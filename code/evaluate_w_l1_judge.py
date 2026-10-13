#!/usr/bin/env python3
"""Multimodal model-judge evaluation for existing ActGround W/L1 outputs.

This script never regenerates W, L1, or frames. It reads cached frame files,
sends them to the OpenAI Responses API, validates structured judgments, and
appends completed requests to a resumable JSONL cache.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


SCRIPT_VERSION = "w_l1_visual_judge_v1"
PROMPT_VERSION = "w_l1_visual_judge_prompt_v1"
DEFAULT_MODEL = "gpt-5.4-2026-03-05"
VIDEO_ID = "P01-20240202-161948"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / VIDEO_ID / "qwen30b_rerun_v2"
DEFAULT_W = DEFAULT_RUN_DIR / "w_outputs.jsonl"
DEFAULT_L1 = DEFAULT_RUN_DIR / "l1_candidates_wgroup_qwen.jsonl"
DEFAULT_GT = DEFAULT_RUN_DIR / "gt.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "judge_gpt5_4"

VERDICTS = ("correct", "partly_correct", "incorrect", "uncertain")
COVERAGE = ("covered", "partly_covered", "missed", "cannot_be_judged")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
REVIEW_FIELDS = (
    "request_id",
    "check_type",
    "subject_id",
    "t_start",
    "t_end",
    "prediction_or_reference",
    "overlapping_gt_narrations",
    "neighboring_l1_descriptions",
    "status",
    "verdict",
    "reference_coverage",
    "explanation",
    "supporting_frame_ids",
    "model",
    "response_id",
    "judgment_label",
)

JUDGE_INSTRUCTIONS = """You are a careful visual judge of egocentric hand-object actions.

The supplied images are the only visual evidence. Predictions, reference
narrations, and neighboring model descriptions are textual claims or hints;
they are not visual evidence. Never claim that an action is visible merely
because a prediction or narration says it happened.

Accept equivalent action and object descriptions rather than requiring exact
wording. Judge the actual action, manipulated object, meaningful state change,
and temporal scope visible in the supplied images. Reference narrations are
useful but can be incomplete. Do not assume an unmatched prediction is wrong.
Prior/nearby context may clarify identity or motion, but an action visible only
in context does not establish that it occurred in the target/event interval.

Use verdicts as follows:
- correct: the prediction/reference coverage is visually supported in its
  important action, object, and temporal scope.
- partly_correct: the core is supported but an important action, object,
  attribute, split, merge, or temporal detail is incomplete or wrong.
- incorrect: the central action/object claim is contradicted or unsupported.
- uncertain: the supplied images do not permit a reliable decision.

The explanation must be brief. Every supporting_frame_id must be copied
exactly from a supplied FRAME_ID and must point to the frames most relevant to
the judgment. Model-generated explanations are judgments, not visual evidence;
the frame IDs are pointers to the underlying evidence for human review.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def interval_overlaps(left: dict, right: dict) -> bool:
    return min(float(left["t_end"]), float(right["t_end"])) > max(
        float(left["t_start"]), float(right["t_start"])
    )


def in_selection(item: dict, start: float, end: float) -> bool:
    return float(item["t_end"]) > start and float(item["t_start"]) < end


def object_name(item: dict) -> str:
    rich = item.get("direct_object_rich")
    if isinstance(rich, dict) and rich.get("name"):
        return str(rich["name"])
    value = item.get("direct_object")
    if isinstance(value, dict):
        value = value.get("name")
    return str(value or "unknown")


def action_text(item: dict) -> str:
    interaction = item.get("interaction") if isinstance(item.get("interaction"), dict) else {}
    details = [str(item.get("verb") or "unknown"), object_name(item)]
    hand = interaction.get("hand") or item.get("hand")
    role = interaction.get("role")
    suffix = ", ".join(str(value) for value in (hand, role) if value)
    return " ".join(details) + (f" [{suffix}]" if suffix else "")


def l1_text(item: dict) -> str:
    label = f"{item.get('verb', 'unknown')} {object_name(item)}"
    description = str(item.get("micro_action_description") or "").strip()
    return f"{label}: {description}" if description else label


def gt_hint(item: dict) -> dict:
    return {
        "gt_id": str(item["gt_id"]),
        "t_start": float(item["t_start"]),
        "t_end": float(item["t_end"]),
        "verb": str(item.get("verb") or ""),
        "direct_object": object_name(item),
        "narration": str(item.get("narration") or "").strip(),
    }


def validate_artifacts(w_rows: list[dict], l1_rows: list[dict], gt_rows: list[dict]) -> dict:
    required = {
        "W": {"window_idx", "t_start", "t_end", "valid_actions"},
        "L1": {"l1_id", "t_start", "t_end", "verb", "direct_object"},
        "GT": {"gt_id", "video_id", "t_start", "t_end", "verb", "direct_object", "narration"},
    }
    for label, rows in (("W", w_rows), ("L1", l1_rows), ("GT", gt_rows)):
        if not rows:
            raise ValueError(f"{label} artifact is empty")
        for index, row in enumerate(rows):
            missing = required[label] - set(row)
            if missing:
                raise ValueError(f"{label} row {index} missing keys: {sorted(missing)}")
            start, end = float(row["t_start"]), float(row["t_end"])
            if not (0 <= start < end):
                raise ValueError(f"{label} row {index} has invalid interval [{start}, {end})")

    w_by_idx: dict[int, dict] = {}
    accepted_refs: set[tuple[int, int]] = set()
    for row in w_rows:
        window_idx = int(row["window_idx"])
        if window_idx in w_by_idx:
            raise ValueError(f"Duplicate W window_idx: {window_idx}")
        if not isinstance(row["valid_actions"], list):
            raise ValueError(f"W window {window_idx}: valid_actions is not a list")
        w_by_idx[window_idx] = row
        for action in row["valid_actions"]:
            if not isinstance(action, dict) or "action_idx" not in action:
                raise ValueError(f"W window {window_idx}: accepted action lacks action_idx")
            accepted_refs.add((window_idx, int(action["action_idx"])))

    l1_ids: set[str] = set()
    for row in l1_rows:
        l1_id = str(row["l1_id"])
        if l1_id in l1_ids:
            raise ValueError(f"Duplicate L1 l1_id: {l1_id}")
        l1_ids.add(l1_id)
        for window_idx in row.get("source_window_idxs", []):
            if int(window_idx) not in w_by_idx:
                raise ValueError(f"L1 {l1_id} references missing W window {window_idx}")
        for ref in row.get("source_action_refs", []):
            key = (int(ref["window_idx"]), int(ref["action_idx"]))
            if key not in accepted_refs:
                raise ValueError(f"L1 {l1_id} references missing accepted W action {key}")

    gt_ids: set[str] = set()
    for row in gt_rows:
        gt_id = str(row["gt_id"])
        if gt_id in gt_ids:
            raise ValueError(f"Duplicate GT gt_id: {gt_id}")
        gt_ids.add(gt_id)
        if str(row["video_id"]) != VIDEO_ID:
            raise ValueError(f"GT {gt_id} belongs to {row['video_id']}, expected {VIDEO_ID}")

    return {
        "W": {"rows": len(w_rows), "keys": sorted({key for row in w_rows for key in row})},
        "L1": {"rows": len(l1_rows), "keys": sorted({key for row in l1_rows for key in row})},
        "GT": {"rows": len(gt_rows), "keys": sorted({key for row in gt_rows for key in row})},
    }


def evenly_sample(items: list[Any], maximum: int) -> list[Any]:
    if maximum <= 0 or not items:
        return []
    if len(items) <= maximum:
        return items
    if maximum == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (maximum - 1)) for index in range(maximum)]
    return [items[index] for index in dict.fromkeys(indices)]


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def safe_id(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


class FrameCatalog:
    """Resolve existing cached W frame groups without extracting new frames."""

    def __init__(self, video_run_root: Path, w_by_idx: dict[int, dict]) -> None:
        self.video_run_root = video_run_root.resolve()
        self.w_by_idx = w_by_idx
        self.fallback_dirs: dict[int, list[Path]] = defaultdict(list)
        self._resolved: dict[int, dict[str, Any] | None] = {}
        self._sha256: dict[Path, str] = {}
        for frames_root in sorted(self.video_run_root.glob("**/frames")):
            if not frames_root.is_dir():
                continue
            for win_dir in sorted(frames_root.glob("win_*")):
                match = re.fullmatch(r"win_(\d+)", win_dir.name)
                if match and win_dir.is_dir():
                    self.fallback_dirs[int(match.group(1))].append(win_dir.resolve())

    def frame_roots(self) -> list[str]:
        return sorted({str(path.parent) for paths in self.fallback_dirs.values() for path in paths})

    def resolve_window(self, window_idx: int) -> dict[str, Any] | None:
        if window_idx in self._resolved:
            return self._resolved[window_idx]
        record = self.w_by_idx.get(window_idx, {})
        candidates: list[Path] = []
        for value in (record.get("frames_path"),):
            if value:
                candidates.append(Path(str(value)).expanduser())
        for key in ("target_frames_path", "before_context_frames_path", "after_context_frames_path"):
            value = record.get(key)
            if value:
                candidates.append(Path(str(value)).expanduser().parent)
        candidates.extend(self.fallback_dirs.get(window_idx, []))
        unique: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        for win_dir in unique:
            target = image_files(win_dir / "target")
            if not target:
                continue
            result = {
                "window_dir": win_dir,
                "target": target,
                "before": image_files(win_dir / "before"),
                "after": image_files(win_dir / "after"),
            }
            self._resolved[window_idx] = result
            return result
        self._resolved[window_idx] = None
        return None

    def file_hash(self, path: Path) -> str:
        path = path.resolve()
        if path not in self._sha256:
            self._sha256[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._sha256[path]

    def descriptor(
        self,
        *,
        frame_id: str,
        role: str,
        path: Path,
        window_idx: int,
        temporal_relation: str,
    ) -> dict:
        return {
            "frame_id": frame_id,
            "role": role,
            "window_idx": int(window_idx),
            "temporal_relation": temporal_relation,
            "path": str(path.resolve()),
            "sha256": self.file_hash(path),
        }


def window_frames(
    catalog: FrameCatalog,
    window_idx: int,
    prefix: str,
    max_target: int,
    max_context: int,
) -> tuple[list[dict], str | None]:
    groups = catalog.resolve_window(window_idx)
    if groups is None:
        return [], f"no cached target frames found for W window {window_idx}"
    frames: list[dict] = []
    for index, path in enumerate(evenly_sample(groups["before"], max_context)):
        frames.append(
            catalog.descriptor(
                frame_id=f"{prefix}:prior:{index:02d}",
                role="prior_context",
                path=path,
                window_idx=window_idx,
                temporal_relation="before_target",
            )
        )
    for index, path in enumerate(evenly_sample(groups["target"], max_target)):
        frames.append(
            catalog.descriptor(
                frame_id=f"{prefix}:target:{index:02d}",
                role="target",
                path=path,
                window_idx=window_idx,
                temporal_relation="inside_target",
            )
        )
    return frames, None


def event_frames(
    catalog: FrameCatalog,
    covering_windows: list[int],
    prefix: str,
    max_event: int,
    max_context: int,
) -> tuple[list[dict], str | None]:
    covering_windows = sorted(set(covering_windows))
    event_candidates: list[tuple[int, Path]] = []
    missing_windows: list[int] = []
    for window_idx in covering_windows:
        groups = catalog.resolve_window(window_idx)
        if groups is None:
            missing_windows.append(window_idx)
            continue
        event_candidates.extend((window_idx, path) for path in groups["target"])
    if not event_candidates:
        interval = f"{covering_windows[0]}..{covering_windows[-1]}" if covering_windows else "none"
        return [], f"no cached event frames found for covering W windows {interval}"

    frames: list[dict] = []
    for index, (window_idx, path) in enumerate(evenly_sample(event_candidates, max_event)):
        frames.append(
            catalog.descriptor(
                frame_id=f"{prefix}:event:{index:02d}",
                role="event",
                path=path,
                window_idx=window_idx,
                temporal_relation="inside_event",
            )
        )

    context_candidates: list[tuple[int, Path, str]] = []
    first_idx, last_idx = covering_windows[0], covering_windows[-1]
    first_groups = catalog.resolve_window(first_idx)
    if first_groups:
        context_candidates.extend((first_idx, path, "before_event") for path in first_groups["before"])
    previous_groups = catalog.resolve_window(first_idx - 1)
    if previous_groups:
        context_candidates.extend((first_idx - 1, path, "before_event") for path in previous_groups["target"])
    next_groups = catalog.resolve_window(last_idx + 1)
    if next_groups:
        context_candidates.extend((last_idx + 1, path, "after_event") for path in next_groups["target"])

    for index, (window_idx, path, relation) in enumerate(evenly_sample(context_candidates, max_context)):
        frames.append(
            catalog.descriptor(
                frame_id=f"{prefix}:context:{index:02d}",
                role="nearby_context",
                path=path,
                window_idx=window_idx,
                temporal_relation=relation,
            )
        )
    note = None
    if missing_windows:
        note = f"cached frames missing for some covering windows: {missing_windows}"
    return frames, note


def overlapping(items: Iterable[dict], subject: dict) -> list[dict]:
    return [item for item in items if interval_overlaps(item, subject)]


def covering_window_indices(subject: dict, w_rows: list[dict]) -> list[int]:
    return [int(row["window_idx"]) for row in w_rows if interval_overlaps(row, subject)]


def make_request_hash(plan: dict, model: str) -> str:
    material = {
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "request_id": plan["request_id"],
        "check_type": plan["check_type"],
        "subject_id": plan["subject_id"],
        "input": plan["input"],
        "frames": plan["frames"],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def finalize_plan(plan: dict, model: str) -> dict:
    plan["request_hash"] = make_request_hash(plan, model)
    return plan


def build_w_plans(
    selected: list[dict],
    gt_rows: list[dict],
    catalog: FrameCatalog,
    model: str,
    max_target: int,
    max_context: int,
) -> list[dict]:
    plans = []
    for row in selected:
        window_idx = int(row["window_idx"])
        subject_id = f"w_{window_idx:06d}"
        frames, blocked = window_frames(
            catalog, window_idx, subject_id, max_target, max_context
        )
        gt = [gt_hint(item) for item in overlapping(gt_rows, row)]
        accepted = row.get("valid_actions") or []
        prompt = (
            "CHECK TYPE: W correctness\n"
            f"Expected request_id: w:{window_idx}\n"
            f"Expected subject_id: {subject_id}\n"
            f"TARGET interval: [{float(row['t_start']):.3f}, {float(row['t_end']):.3f}) seconds.\n"
            "Judge whether the complete set of accepted W predictions correctly describes actions "
            "visibly occurring in TARGET frames. The PRIOR CONTEXT frames are identity/transition "
            "context only. Consider missing visible actions and extra unsupported actions. Do not "
            "reject a prediction merely because no narration matches it.\n\n"
            f"Accepted W prediction claims:\n{json.dumps(accepted, ensure_ascii=False, indent=2)}\n\n"
            "Overlapping GT narration hints (potentially incomplete):\n"
            f"{json.dumps(gt, ensure_ascii=False, indent=2)}\n"
            "For this check, reference_coverage must be not_applicable."
        )
        plans.append(
            finalize_plan(
                {
                    "request_id": f"w:{window_idx}",
                    "check_type": "w_correctness",
                    "subject_id": subject_id,
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "input": {
                        "accepted_action_predictions": accepted,
                        "overlapping_gt_narrations": gt,
                        "frame_scope_note": "target frames judged; prior context is context only",
                    },
                    "frames": frames,
                    "prompt": prompt,
                    "blocked_reason": blocked,
                    "prediction_or_reference": "; ".join(action_text(item) for item in accepted)
                    or "(no accepted W action)",
                    "gt_text": "; ".join(item["narration"] for item in gt),
                    "neighbor_text": "",
                },
                model,
            )
        )
    return plans


def neighboring_l1(l1_rows: list[dict], index: int, count: int = 2) -> list[dict]:
    neighbors = []
    for neighbor_index in range(max(0, index - count), min(len(l1_rows), index + count + 1)):
        if neighbor_index == index:
            continue
        row = l1_rows[neighbor_index]
        neighbors.append(
            {
                "l1_id": str(row["l1_id"]),
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "description": l1_text(row),
            }
        )
    return neighbors


def build_l1_plans(
    selected_ids: set[str],
    l1_rows: list[dict],
    w_rows: list[dict],
    gt_rows: list[dict],
    catalog: FrameCatalog,
    model: str,
    max_event: int,
    max_context: int,
) -> list[dict]:
    plans = []
    for index, row in enumerate(l1_rows):
        l1_id = str(row["l1_id"])
        if l1_id not in selected_ids:
            continue
        source_windows = [int(value) for value in row.get("source_window_idxs", [])]
        cover = source_windows or covering_window_indices(row, w_rows)
        subject_id = safe_id(l1_id)
        frames, frame_note = event_frames(
            catalog, cover, subject_id, max_event, max_context
        )
        blocked = None if frames else frame_note
        gt = [gt_hint(item) for item in overlapping(gt_rows, row)]
        neighbors = neighboring_l1(l1_rows, index)
        prompt = (
            "CHECK TYPE: L1 correctness\n"
            f"Expected request_id: l1:{l1_id}\n"
            f"Expected subject_id: {subject_id}\n"
            f"L1 interval: [{float(row['t_start']):.3f}, {float(row['t_end']):.3f}) seconds.\n"
            "Judge the L1 prediction against EVENT frames. NEARBY CONTEXT helps with transitions "
            "but is not itself event evidence. Also assess whether the event is incorrectly split "
            "from or merged with neighboring L1 events. Neighbor descriptions are model claims, "
            "not visual evidence. Do not reject a prediction merely because GT has no match.\n\n"
            f"L1 prediction claim:\n{json.dumps(row, ensure_ascii=False, indent=2)}\n\n"
            f"Neighboring L1 claims:\n{json.dumps(neighbors, ensure_ascii=False, indent=2)}\n\n"
            "Overlapping GT narration hints (potentially incomplete):\n"
            f"{json.dumps(gt, ensure_ascii=False, indent=2)}\n"
            "For this check, reference_coverage must be not_applicable."
        )
        plans.append(
            finalize_plan(
                {
                    "request_id": f"l1:{l1_id}",
                    "check_type": "l1_correctness",
                    "subject_id": subject_id,
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "input": {
                        "l1_prediction": row,
                        "neighboring_l1_predictions": neighbors,
                        "overlapping_gt_narrations": gt,
                        "covering_window_idxs": cover,
                        "frame_scope_note": "event frames judged; nearby context is context only",
                        "frame_availability_note": frame_note,
                    },
                    "frames": frames,
                    "prompt": prompt,
                    "blocked_reason": blocked,
                    "prediction_or_reference": l1_text(row),
                    "gt_text": "; ".join(item["narration"] for item in gt),
                    "neighbor_text": "; ".join(item["description"] for item in neighbors),
                },
                model,
            )
        )
    return plans


def build_gt_plans(
    selected: list[dict],
    l1_rows: list[dict],
    w_rows: list[dict],
    catalog: FrameCatalog,
    model: str,
    max_event: int,
    max_context: int,
) -> list[dict]:
    plans = []
    for row in selected:
        gt_id = str(row["gt_id"])
        subject_id = safe_id(gt_id)
        cover = covering_window_indices(row, w_rows)
        frames, frame_note = event_frames(
            catalog, cover, subject_id, max_event, max_context
        )
        blocked = None if frames else frame_note
        l1_matches = [
            {
                "l1_id": str(item["l1_id"]),
                "t_start": float(item["t_start"]),
                "t_end": float(item["t_end"]),
                "description": l1_text(item),
            }
            for item in overlapping(l1_rows, row)
        ]
        reference = gt_hint(row)
        prompt = (
            "CHECK TYPE: missing L1/reference coverage\n"
            f"Expected request_id: gt:{gt_id}\n"
            f"Expected subject_id: {subject_id}\n"
            f"Reference interval: [{float(row['t_start']):.3f}, {float(row['t_end']):.3f}) seconds.\n"
            "Independently decide whether the reference action is covered by one or more overlapping "
            "L1 predictions, partly covered, missed, or cannot be judged from the EVENT images. "
            "The reference narration is useful but potentially incomplete. An empty overlapping-L1 "
            "list must still be visually judged; do not infer a miss from string matching alone.\n\n"
            f"Reference event hint:\n{json.dumps(reference, ensure_ascii=False, indent=2)}\n\n"
            f"Overlapping L1 prediction claims (may be empty):\n"
            f"{json.dumps(l1_matches, ensure_ascii=False, indent=2)}\n\n"
            "Set verdict consistently: covered=correct, partly_covered=partly_correct, "
            "missed=incorrect, cannot_be_judged=uncertain."
        )
        plans.append(
            finalize_plan(
                {
                    "request_id": f"gt:{gt_id}",
                    "check_type": "missing_l1_reference_coverage",
                    "subject_id": subject_id,
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "input": {
                        "reference_event": reference,
                        "overlapping_l1_predictions": l1_matches,
                        "covering_window_idxs": cover,
                        "frame_scope_note": "event frames judged; nearby context is context only",
                        "frame_availability_note": frame_note,
                    },
                    "frames": frames,
                    "prompt": prompt,
                    "blocked_reason": blocked,
                    "prediction_or_reference": (
                        f"REFERENCE {reference['verb']} {reference['direct_object']}: "
                        f"{reference['narration']} | OVERLAPPING L1: "
                        + ("; ".join(item["description"] for item in l1_matches) or "(none)")
                    ),
                    "gt_text": reference["narration"],
                    "neighbor_text": "; ".join(item["description"] for item in l1_matches),
                },
                model,
            )
        )
    return plans


def result_schema(plan: dict) -> dict:
    coverage_values = (
        list(COVERAGE)
        if plan["check_type"] == "missing_l1_reference_coverage"
        else ["not_applicable"]
    )
    frame_ids = [frame["frame_id"] for frame in plan["frames"]]
    return {
        "type": "object",
        "properties": {
            "request_id": {"type": "string", "enum": [plan["request_id"]]},
            "check_type": {"type": "string", "enum": [plan["check_type"]]},
            "subject_id": {"type": "string", "enum": [plan["subject_id"]]},
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "reference_coverage": {"type": "string", "enum": coverage_values},
            "explanation": {"type": "string"},
            "supporting_frame_ids": {
                "type": "array",
                "items": {"type": "string", "enum": frame_ids},
            },
        },
        "required": [
            "request_id",
            "check_type",
            "subject_id",
            "verdict",
            "reference_coverage",
            "explanation",
            "supporting_frame_ids",
        ],
        "additionalProperties": False,
    }


def validate_result(result: dict, plan: dict) -> None:
    expected_keys = {
        "request_id",
        "check_type",
        "subject_id",
        "verdict",
        "reference_coverage",
        "explanation",
        "supporting_frame_ids",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise ValueError(f"structured result keys do not match schema: {sorted(result) if isinstance(result, dict) else type(result)}")
    for field in ("request_id", "check_type", "subject_id"):
        if result[field] != plan[field]:
            raise ValueError(f"returned {field}={result[field]!r}, expected {plan[field]!r}")
    if result["verdict"] not in VERDICTS:
        raise ValueError(f"invalid verdict: {result['verdict']!r}")
    expected_coverage = (
        set(COVERAGE)
        if plan["check_type"] == "missing_l1_reference_coverage"
        else {"not_applicable"}
    )
    if result["reference_coverage"] not in expected_coverage:
        raise ValueError(f"invalid reference_coverage: {result['reference_coverage']!r}")
    if not isinstance(result["explanation"], str) or not result["explanation"].strip():
        raise ValueError("explanation must be a non-empty string")
    returned_ids = result["supporting_frame_ids"]
    if not isinstance(returned_ids, list) or not returned_ids:
        raise ValueError("supporting_frame_ids must be a non-empty list")
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("supporting_frame_ids contains duplicates")
    valid_ids = {frame["frame_id"] for frame in plan["frames"]}
    invalid = [frame_id for frame_id in returned_ids if frame_id not in valid_ids]
    if invalid:
        raise ValueError(f"returned unknown supporting frame IDs: {invalid}")
    if plan["check_type"] == "missing_l1_reference_coverage":
        mapping = {
            "covered": "correct",
            "partly_covered": "partly_correct",
            "missed": "incorrect",
            "cannot_be_judged": "uncertain",
        }
        expected_verdict = mapping[result["reference_coverage"]]
        if result["verdict"] != expected_verdict:
            raise ValueError(
                f"coverage {result['reference_coverage']} requires verdict {expected_verdict}, "
                f"got {result['verdict']}"
            )


def mime_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image type: {path}")
    return guessed


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type(path)};base64,{encoded}"


def response_content(plan: dict, image_detail: str) -> list[dict]:
    content: list[dict] = [{"type": "input_text", "text": plan["prompt"]}]
    for frame in plan["frames"]:
        role = frame["role"]
        if role in {"prior_context", "nearby_context"}:
            warning = "CONTEXT ONLY; do not count an action visible only here as target/event evidence."
        else:
            warning = "PRIMARY VISUAL EVIDENCE for the target/event interval."
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"FRAME_ID={frame['frame_id']} ROLE={role} "
                    f"WINDOW={frame['window_idx']} RELATION={frame['temporal_relation']}. {warning}"
                ),
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": image_data_url(Path(frame["path"])),
                "detail": image_detail,
            }
        )
    return content


def call_judge(client: Any, plan: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    schema = result_schema(plan)
    retryable_names = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
    for attempt in range(args.max_retries + 1):
        try:
            response = client.responses.create(
                model=args.model,
                instructions=JUDGE_INSTRUCTIONS,
                input=[{"role": "user", "content": response_content(plan, args.image_detail)}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "actground_visual_judgment",
                        "description": "A validated visual action judgment; not official HD-EPIC accuracy.",
                        "strict": True,
                        "schema": schema,
                    },
                    "verbosity": "low",
                },
                reasoning={"effort": args.reasoning_effort},
                max_output_tokens=args.max_output_tokens,
                store=False,
                metadata={
                    "video_id": VIDEO_ID,
                    "check_type": plan["check_type"],
                    "subject_id": plan["subject_id"][:512],
                    "script_version": SCRIPT_VERSION,
                },
            )
            if response.status != "completed":
                raise RuntimeError(
                    f"response {response.id} status={response.status} "
                    f"incomplete_details={getattr(response, 'incomplete_details', None)}"
                )
            output_text = response.output_text
            if not output_text:
                raise RuntimeError(f"response {response.id} has no output_text")
            try:
                result = json.loads(output_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"response {response.id} returned invalid JSON: {exc}") from exc
            validate_result(result, plan)
            usage = getattr(response, "usage", None)
            response_meta = {
                "response_id": response.id,
                "response_model": response.model,
                "response_status": response.status,
                "created_at": getattr(response, "created_at", None),
                "usage": usage.model_dump(mode="json") if usage is not None else None,
            }
            return result, response_meta
        except Exception as exc:
            retryable = exc.__class__.__name__ in retryable_names
            if not retryable or attempt >= args.max_retries:
                raise
            wait_seconds = min(60, 2 ** attempt * 2)
            print(
                f"Retryable API error for {plan['request_id']} "
                f"({exc.__class__.__name__}); retrying in {wait_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def validate_cached_row(row: dict, source: str) -> None:
    required = {
        "request_id",
        "request_hash",
        "check_type",
        "subject_id",
        "model",
        "input",
        "frames",
        "result",
        "response",
        "judgment_label",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"{source}: cached judgment missing keys: {sorted(missing)}")
    pseudo_plan = {
        "request_id": row["request_id"],
        "check_type": row["check_type"],
        "subject_id": row["subject_id"],
        "frames": row["frames"],
    }
    validate_result(row["result"], pseudo_plan)


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for index, row in enumerate(read_jsonl(path), 1):
        validate_cached_row(row, f"{path}:{index}")
        request_hash = str(row["request_hash"])
        if request_hash in cache and cache[request_hash] != row:
            raise ValueError(f"{path}:{index}: conflicting duplicate request_hash {request_hash}")
        cache[request_hash] = row
    return cache


def append_judgment(path: Path, row: dict) -> None:
    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def judgment_row(plan: dict, model: str, result: dict, response_meta: dict) -> dict:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at": utc_now(),
        "judgment_label": "model_judgment_not_official_hdepic_accuracy",
        "video_id": VIDEO_ID,
        "request_id": plan["request_id"],
        "request_hash": plan["request_hash"],
        "check_type": plan["check_type"],
        "subject_id": plan["subject_id"],
        "t_start": plan["t_start"],
        "t_end": plan["t_end"],
        "model": model,
        "input": plan["input"],
        "input_prompt": plan["prompt"],
        "frames": plan["frames"],
        "result": result,
        "response": response_meta,
    }


def review_rows(plans: list[dict], cache: dict[str, dict], dry_run: bool) -> list[dict]:
    rows = []
    for plan in plans:
        cached = cache.get(plan["request_hash"])
        result = cached["result"] if cached else {}
        if cached:
            status = "cached_model_judgment"
        elif plan["blocked_reason"]:
            status = "not_judged_missing_cached_frames"
        elif dry_run:
            status = "dry_run_ready"
        else:
            status = "pending"
        rows.append(
            {
                "request_id": plan["request_id"],
                "check_type": plan["check_type"],
                "subject_id": plan["subject_id"],
                "t_start": plan["t_start"],
                "t_end": plan["t_end"],
                "prediction_or_reference": plan["prediction_or_reference"],
                "overlapping_gt_narrations": plan["gt_text"],
                "neighboring_l1_descriptions": plan["neighbor_text"],
                "status": status,
                "verdict": result.get("verdict", ""),
                "reference_coverage": result.get("reference_coverage", ""),
                "explanation": result.get("explanation", plan["blocked_reason"] or ""),
                "supporting_frame_ids": "; ".join(result.get("supporting_frame_ids", [])),
                "model": cached.get("model", "") if cached else "",
                "response_id": cached.get("response", {}).get("response_id", "") if cached else "",
                "judgment_label": cached.get("judgment_label", "") if cached else "",
            }
        )
    return rows


def write_review(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_summary(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    schemas: dict,
    catalog: FrameCatalog,
    plans: list[dict],
    cache: dict[str, dict],
    initial_cache_size: int,
    requests_made: int,
    run_error: str | None,
) -> dict:
    planned = Counter(plan["check_type"] for plan in plans)
    blocked = Counter(plan["check_type"] for plan in plans if plan["blocked_reason"])
    hits = Counter(plan["check_type"] for plan in plans if plan["request_hash"] in cache)
    pending = Counter(
        plan["check_type"]
        for plan in plans
        if not plan["blocked_reason"] and plan["request_hash"] not in cache
    )
    verdicts: dict[str, Counter] = defaultdict(Counter)
    coverage = Counter()
    for plan in plans:
        cached = cache.get(plan["request_hash"])
        if not cached:
            continue
        result = cached["result"]
        verdicts[plan["check_type"]][result["verdict"]] += 1
        if plan["check_type"] == "missing_l1_reference_coverage":
            coverage[result["reference_coverage"]] += 1
    selected_gt = planned.get("missing_l1_reference_coverage", 0)
    judged_gt = sum(coverage.values())
    return {
        "created_at": utc_now(),
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluation_label": "model_judgments_not_official_hdepic_accuracy",
        "official_hdepic_accuracy": False,
        "dry_run": bool(args.dry_run),
        "model": args.model,
        "video_id": VIDEO_ID,
        "selection": {"start_inclusive": args.start, "end_exclusive": args.end},
        "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "api_requests_made_this_run": requests_made,
        "cache_entries_before": initial_cache_size,
        "cache_entries_after": len(cache),
        "inputs": {name: str(path.resolve()) for name, path in paths.items()},
        "observed_artifact_schemas": schemas,
        "cached_frame_roots_found": catalog.frame_roots(),
        "counts": {
            "planned": dict(planned),
            "blocked_missing_cached_frames": dict(blocked),
            "completed_or_cached": dict(hits),
            "pending_ready_for_api": dict(pending),
        },
        "verdict_counts": {check: dict(counts) for check, counts in verdicts.items()},
        "reference_coverage": {
            "selected_reference_events": selected_gt,
            "judged_reference_events": judged_gt,
            "not_judged": selected_gt - judged_gt,
            "counts": dict(coverage),
        },
        "blocked_subjects": [
            {
                "request_id": plan["request_id"],
                "check_type": plan["check_type"],
                "subject_id": plan["subject_id"],
                "reason": plan["blocked_reason"],
            }
            for plan in plans
            if plan["blocked_reason"]
        ],
        "uncertain_cases_remain_in_outputs": True,
        "run_error": run_error,
    }


def write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w-jsonl", type=Path, default=DEFAULT_W)
    parser.add_argument("--l1-jsonl", type=Path, default=DEFAULT_L1)
    parser.add_argument("--gt-jsonl", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=float, default=0.0, help="Inclusive start time in seconds.")
    parser.add_argument("--end", type=float, default=770.0, help="Exclusive end time in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and validate without API calls.")
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh"), default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-w-target-frames", type=int, default=8)
    parser.add_argument("--max-w-context-frames", type=int, default=6)
    parser.add_argument("--max-l1-event-frames", type=int, default=24)
    parser.add_argument("--max-l1-context-frames", type=int, default=6)
    parser.add_argument("--max-gt-event-frames", type=int, default=24)
    parser.add_argument("--max-gt-context-frames", type=int, default=6)
    args = parser.parse_args(argv)
    if not (0 <= args.start < args.end):
        parser.error("require 0 <= --start < --end")
    for name in (
        "max_output_tokens",
        "max_w_target_frames",
        "max_l1_event_frames",
        "max_gt_event_frames",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "max_retries",
        "max_w_context_frames",
        "max_l1_context_frames",
        "max_gt_context_frames",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "w_jsonl": args.w_jsonl,
        "l1_jsonl": args.l1_jsonl,
        "gt_jsonl": args.gt_jsonl,
    }
    w_rows = read_jsonl(args.w_jsonl)
    l1_rows = read_jsonl(args.l1_jsonl)
    gt_rows = read_jsonl(args.gt_jsonl)
    schemas = validate_artifacts(w_rows, l1_rows, gt_rows)
    w_by_idx = {int(row["window_idx"]): row for row in w_rows}
    video_run_root = args.w_jsonl.resolve().parent.parent
    if video_run_root.name != VIDEO_ID:
        video_run_root = PROJECT_ROOT / "runs" / VIDEO_ID
    catalog = FrameCatalog(video_run_root, w_by_idx)

    selected_w = [row for row in w_rows if in_selection(row, args.start, args.end)]
    selected_l1 = [row for row in l1_rows if in_selection(row, args.start, args.end)]
    selected_gt = [row for row in gt_rows if in_selection(row, args.start, args.end)]
    plans = build_w_plans(
        selected_w,
        gt_rows,
        catalog,
        args.model,
        args.max_w_target_frames,
        args.max_w_context_frames,
    )
    plans.extend(
        build_l1_plans(
            {str(row["l1_id"]) for row in selected_l1},
            l1_rows,
            w_rows,
            gt_rows,
            catalog,
            args.model,
            args.max_l1_event_frames,
            args.max_l1_context_frames,
        )
    )
    plans.extend(
        build_gt_plans(
            selected_gt,
            l1_rows,
            w_rows,
            catalog,
            args.model,
            args.max_gt_event_frames,
            args.max_gt_context_frames,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = args.output_dir / "judgments.jsonl"
    review_path = args.output_dir / "review.csv"
    summary_path = args.output_dir / "summary.json"
    judgments_path.touch(exist_ok=True)
    cache = load_cache(judgments_path)
    initial_cache_size = len(cache)
    requests_made = 0
    run_error: str | None = None

    def save_outputs() -> None:
        write_review(review_path, review_rows(plans, cache, args.dry_run))
        summary = build_summary(
            args=args,
            paths=paths,
            schemas=schemas,
            catalog=catalog,
            plans=plans,
            cache=cache,
            initial_cache_size=initial_cache_size,
            requests_made=requests_made,
            run_error=run_error,
        )
        write_summary(summary_path, summary)

    planned_counts = Counter(plan["check_type"] for plan in plans)
    blocked_plans = [plan for plan in plans if plan["blocked_reason"]]
    pending_plans = [
        plan
        for plan in plans
        if not plan["blocked_reason"] and plan["request_hash"] not in cache
    ]
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "range": [args.start, args.end],
                "model": args.model,
                "planned": dict(planned_counts),
                "blocked_missing_cached_frames": len(blocked_plans),
                "cache_hits": len(plans) - len(pending_plans) - len(blocked_plans),
                "pending_api_requests": len(pending_plans),
                "outputs": {
                    "judgments": str(judgments_path.resolve()),
                    "review": str(review_path.resolve()),
                    "summary": str(summary_path.resolve()),
                },
            },
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        save_outputs()
        return 0
    if blocked_plans:
        run_error = (
            f"Refusing partial evaluation: {len(blocked_plans)} selected requests lack cached "
            "event/target frames. No API requests were made."
        )
        save_outputs()
        print(f"ERROR: {run_error}", file=sys.stderr)
        return 2
    if not pending_plans:
        save_outputs()
        print("All selected requests were already cached.", flush=True)
        return 0
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        run_error = "OPENAI_API_KEY is not configured; no API requests were made."
        save_outputs()
        print(f"ERROR: {run_error}", file=sys.stderr)
        return 3

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=args.request_timeout, max_retries=0)
    try:
        for index, plan in enumerate(pending_plans, 1):
            print(
                f"[{index}/{len(pending_plans)}] judging {plan['request_id']} "
                f"with {len(plan['frames'])} cached images",
                flush=True,
            )
            result, response_meta = call_judge(client, plan, args)
            row = judgment_row(plan, args.model, result, response_meta)
            validate_cached_row(row, plan["request_id"])
            append_judgment(judgments_path, row)
            cache[plan["request_hash"]] = row
            requests_made += 1
    except Exception as exc:
        run_error = f"{exc.__class__.__name__}: {exc}"
        print(f"ERROR: {run_error}", file=sys.stderr, flush=True)
    finally:
        save_outputs()
    return 1 if run_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
