#!/usr/bin/env python3
"""Build a strict W-grouping L1 baseline from existing W outputs only.

This script does not run W, VLMs, scene observation, frame extraction, or L1
reasoning. It reads an existing W JSONL file and groups consecutive 1-second W
windows by exact normalized (verb, object) signature in independent per-hand
streams.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_W_JSONL = (
    "/data2/yaxuanli/hmgraph_codex/runs/P01-20240202-110250/full/w_outputs.jsonl"
)
DEFAULT_OUT_JSONL = (
    "/data2/yaxuanli/hmgraph_codex/runs/P01-20240202-110250/full/"
    "l1_candidates_wgroup.jsonl"
)
PREVIOUS_WGROUP_JSONL = "/mnt/user-data/uploads/l1_candidates_wgroup.jsonl"
HANDS = ("left", "right", "both", "unknown")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def norm_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def get_obj_name(action: dict) -> str:
    rich = action.get("direct_object_rich")
    if isinstance(rich, dict) and rich.get("name") is not None:
        return str(rich.get("name"))
    if action.get("direct_object") is not None:
        return str(action.get("direct_object"))
    return ""


def valid_primary_actions(record: dict) -> list[dict]:
    if not bool(record.get("action_present")):
        return []
    actions = record.get("valid_actions") or []
    actions = [action for action in actions if isinstance(action, dict)]
    if not actions:
        return []
    return [
        action
        for action in actions
        if norm_text((action.get("interaction") or {}).get("role")) == "primary_manipulation"
    ]


def action_signature(action: dict) -> tuple[str, str]:
    return norm_text(action.get("verb")), norm_text(get_obj_name(action))


def action_hand(action: dict) -> str:
    hand = norm_text((action.get("interaction") or {}).get("hand"))
    return hand if hand in HANDS else "unknown"


def per_hand_snapshot(record: dict) -> dict[str, dict | None]:
    snapshot: dict[str, dict | None] = {hand: None for hand in HANDS}
    for hand in HANDS:
        candidates = [
            action
            for action in valid_primary_actions(record)
            if action_hand(action) == hand
        ]
        if not candidates:
            continue
        action = max(candidates, key=lambda item: safe_float(item.get("confidence"), 0.0))
        snapshot[hand] = {
            "sig": action_signature(action),
            "confidence": safe_float(action.get("confidence"), 0.0),
            "window_idx": int(record["window_idx"]),
            "action": action,
        }
    return snapshot


def new_group(snapshot_item: dict, hand: str, record: dict) -> dict:
    return {
        "sig": snapshot_item["sig"],
        "t_start": safe_float(record.get("t_start"), 0.0),
        "t_end": safe_float(record.get("t_end"), 0.0),
        "window_idxs": [int(snapshot_item["window_idx"])],
        "confidences": [safe_float(snapshot_item.get("confidence"), 0.0)],
        "hand": hand,
    }


def build_groups(records: list[dict]) -> tuple[list[dict], dict]:
    groups: list[dict] = []
    open_group: dict[str, dict | None] = {hand: None for hand in HANDS}
    primary_windows_by_hand = {hand: 0 for hand in HANDS}
    bimanual_window_idxs: list[int] = []

    def emit(group: dict | None) -> None:
        if group is not None:
            groups.append(group)

    for record in sorted(records, key=lambda r: safe_float(r.get("t_start"), 0.0)):
        snapshot = per_hand_snapshot(record)
        if snapshot["left"] is not None and snapshot["right"] is not None:
            bimanual_window_idxs.append(int(record["window_idx"]))
        for hand in HANDS:
            s = snapshot[hand]
            if s is not None:
                primary_windows_by_hand[hand] += 1
            g = open_group[hand]
            if s is None:
                emit(g)
                open_group[hand] = None
                continue
            if (
                g is not None
                and g["sig"] == s["sig"]
                and int(s["window_idx"]) == int(g["window_idxs"][-1]) + 1
            ):
                g["t_end"] = safe_float(record.get("t_end"), 0.0)
                g["window_idxs"].append(int(s["window_idx"]))
                g["confidences"].append(safe_float(s.get("confidence"), 0.0))
            else:
                emit(g)
                open_group[hand] = new_group(s, hand, record)

    for hand in HANDS:
        emit(open_group[hand])

    stats = {
        "primary_windows_by_hand": primary_windows_by_hand,
        "bimanual_window_idxs": bimanual_window_idxs,
    }
    return groups, stats


def group_to_l1(idx: int, group: dict) -> dict:
    canon_verb, canon_object = group["sig"]
    t_start = safe_float(group.get("t_start"), 0.0)
    t_end = safe_float(group.get("t_end"), 0.0)
    duration = t_end - t_start
    confidence = round(statistics.mean(group["confidences"]), 2)
    n_windows = len(group["window_idxs"])
    hand = str(group["hand"])
    return {
        "l1_id": f"wg_{idx:04d}",
        "segment_id": f"wg_seg_{idx:04d}",
        "t_start": t_start,
        "t_end": t_end,
        "hand": hand,
        "verb": canon_verb,
        "direct_object": {"name": canon_object},
        "micro_action_description": (
            f"From W grouping ({hand} hand): the person {canon_verb} the {canon_object} "
            f"over {duration:.1f}s ({n_windows} consecutive 1s W windows)."
        ),
        "event_subtype": "core_effect",
        "evidence_type": "observed",
        "specificity_level": "from_w",
        "confidence": confidence,
        "confidence_raw_model": confidence,
        "hallucination_risk": "n/a",
        "qa_importance": "unknown",
        "source": "w_grouping_baseline",
        "merged_from_l1_ids": [],
        "supporting_l1_ids": [],
        "related_l1_ids": [],
        "warnings": [],
        "source_window_idxs": [int(window_idx) for window_idx in group["window_idxs"]],
    }


def safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def events_overlapping_window_range(events: list[dict], start_idx: int, end_idx: int) -> list[dict]:
    return [
        event
        for event in events
        if any(start_idx <= int(idx) <= end_idx for idx in event.get("source_window_idxs", []))
    ]


def print_bimanual_moment(records_by_idx: dict[int, dict], events: list[dict], window_idx: int) -> None:
    record = records_by_idx[window_idx]
    print(f"Window {window_idx} valid_actions:")
    print(json.dumps(record.get("valid_actions") or [], ensure_ascii=False, indent=2))
    overlapping = events_overlapping_window_range(events, window_idx, window_idx)
    print(f"L1 events emitted for window {window_idx}:")
    print(json.dumps(overlapping, ensure_ascii=False, indent=2))


def previous_count() -> int | None:
    path = Path(PREVIOUS_WGROUP_JSONL)
    if not path.exists() or not path.is_file():
        return None
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def print_summary(records: list[dict], stats: dict, events: list[dict], out_path: Path) -> None:
    lengths = [len(event["source_window_idxs"]) for event in events]
    signature_counts = Counter(
        (event["verb"], event["direct_object"]["name"]) for event in events
    )
    event_counts_by_hand = Counter(event.get("hand", "unknown") for event in events)
    mean_len = statistics.mean(lengths) if lengths else 0.0
    median_len = statistics.median(lengths) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    records_by_idx = {int(record["window_idx"]): record for record in records}
    bimanual_window_idxs = stats["bimanual_window_idxs"]

    print(f"W windows read: {len(records)}")
    print("W windows with >=1 primary action per hand:")
    for hand in HANDS:
        print(f"  {hand}: {stats['primary_windows_by_hand'][hand]}")
    print(f"L1 events emitted: {len(events)}")
    print("L1 events emitted by hand:")
    for hand in HANDS:
        print(f"  {hand}: {event_counts_by_hand.get(hand, 0)}")
    print(f"Bimanual seconds: {len(bimanual_window_idxs)}")
    if len(bimanual_window_idxs) == 0:
        raise SystemExit(
            "ERROR: bimanual seconds = 0. W is not emitting simultaneous left/right "
            "primary actions even though the prompt was changed."
        )
    print(
        "Group length in windows: "
        f"mean={mean_len:.2f} median={median_len:.2f} max={max_len}"
    )
    print("Top 10 (verb, object) signatures by event count:")
    for (verb, obj), count in signature_counts.most_common(10):
        print(f"  {verb} | {obj}: {count}")
    print("First 5 events:")
    for event in events[:5]:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    print("First 3 bimanual moments:")
    for window_idx in bimanual_window_idxs[:3]:
        print_bimanual_moment(records_by_idx, events, window_idx)
    print("Window_idx around 252: W valid_actions and overlapping L1 events for [251, 254]:")
    for window_idx in range(251, 255):
        if window_idx in records_by_idx:
            print_bimanual_moment(records_by_idx, events, window_idx)
    prior_count = previous_count()
    if prior_count is None:
        print(f"Previous wgroup output not found at {PREVIOUS_WGROUP_JSONL}; old count unavailable.")
    else:
        relation = ">=" if len(events) >= prior_count else "<"
        print(f"Previous wgroup event count: {prior_count}; new count: {len(events)} ({relation} old count)")
    print(f"Output file: {out_path}")
    print(f"Output size bytes: {out_path.stat().st_size if out_path.exists() else 0}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-jsonl", default=DEFAULT_W_JSONL)
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    args = parser.parse_args()

    w_path = Path(args.w_jsonl)
    out_path = Path(args.out_jsonl)
    if not w_path.exists():
        raise SystemExit(
            f"ERROR: W JSONL does not exist: {w_path}\n"
            "Run the W stage first. This script does not regenerate W or call any VLM."
        )
    if not w_path.is_file() or w_path.stat().st_size == 0:
        raise SystemExit(
            f"ERROR: W JSONL is missing or empty: {w_path}\n"
            "Run the W stage first. This script does not regenerate W or call any VLM."
        )

    records = read_jsonl(w_path)
    groups, stats = build_groups(records)
    events = [group_to_l1(idx, group) for idx, group in enumerate(groups, 1)]
    events.sort(key=lambda event: (safe_float(event.get("t_start"), 0.0), str(event.get("hand", ""))))
    for idx, event in enumerate(events, 1):
        event["l1_id"] = f"wg_{idx:04d}"
        event["segment_id"] = f"wg_seg_{idx:04d}"
    write_jsonl(events, out_path)
    print_summary(records, stats, events, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
