"""Build W records for a bounded video range with LLaVAction-7B."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from code.utils.alias_table import build as build_alias_table
from code.utils.hdepic_loader import load_video_actions, write_jsonl
from code.utils.llavaction_vlm import LLaVActionVLM, PROMPT, PROMPT_ID
from code.w.window_extractor import extract_context_frame_groups, make_windows


MODEL_ID = "MLAdaptiveIntelligence/LLaVAction-7B"
MODEL_REVISION = "4b43e5223635402c98e5ba51b07ca1b7a7b96b0a"
VISION_TOWER_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
VALID_HANDS = {"left", "right", "both", "unknown"}
NO_ACTION_TERMS = {"none", "no action", "nothing", "unknown"}
KNOWN_VERBS = sorted(
    {
        "switch off",
        "switch on",
        "plug in",
        "put down",
        "pick up",
        "reach for",
        "take out",
        "take off",
        "pour into",
        "turn off",
        "turn on",
        "open",
        "close",
        "press",
        "pour",
        "stir",
        "mix",
        "grate",
        "cut",
        "slice",
        "peel",
        "write",
        "hold",
        "move",
        "release",
        "place",
        "insert",
        "remove",
        "shake",
        "scrape",
        "transfer",
        "package",
        "hit",
        "wash",
        "clean",
        "wipe",
        "rinse",
        "lift",
        "lower",
        "turn",
        "twist",
        "tilt",
    },
    key=len,
    reverse=True,
)


def _json_object(text: str) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _split_narration(text: str) -> tuple[str, str] | None:
    """Split LLaVAction's native short narration into verb and object."""
    if any(character in text for character in ("{", "}", "[", "]", ":")):
        return None
    narration = _clean_text(text.strip("` \n\t.{}[]\"'"))
    if not narration or narration in NO_ACTION_TERMS or narration.startswith("no action"):
        return None
    for verb in KNOWN_VERBS:
        if narration == verb:
            return verb, "unknown object"
        if narration.startswith(verb + " "):
            return verb, narration[len(verb) + 1 :].strip()
    words = narration.split()
    if len(words) >= 2:
        return words[0], " ".join(words[1:])
    return None


def _strip_object_from_verb(verb: str, object_name: str) -> str:
    """Repair JSON where the model copied the object into the verb field."""
    suffix = " " + object_name
    if object_name and verb.endswith(suffix):
        candidate = verb[: -len(suffix)].strip()
        if candidate:
            verb = candidate
    for known_verb in KNOWN_VERBS:
        if verb == known_verb or verb.startswith(known_verb + " "):
            return known_verb
    return verb


def _normalize_action_phrase(verb: str, object_name: str) -> tuple[str, str]:
    """Recover the affected object when narration leaks into the verb field."""
    for known_verb in KNOWN_VERBS:
        if verb == known_verb:
            return known_verb, object_name
        if verb.startswith(known_verb + " "):
            tail = verb[len(known_verb) + 1 :].strip()
            for separator in (
                " using ",
                " with ",
                " onto ",
                " into ",
                " from ",
                " on ",
                " to ",
            ):
                tail = tail.split(separator, 1)[0].strip()
            return known_verb, tail or object_name
    return _strip_object_from_verb(verb, object_name), object_name


def parse_actions(raw_output: str) -> tuple[list[dict], bool]:
    """Parse compact model JSON and convert it to W-compatible actions."""
    payload = _json_object(raw_output)
    parse_ok = payload is not None and isinstance(payload.get("actions"), list)
    if parse_ok:
        raw_actions = payload["actions"][:2]
    else:
        split = _split_narration(raw_output)
        raw_actions = (
            [
                {
                    "verb": split[0],
                    "direct_object": split[1],
                    "hand": "unknown",
                    "confidence": 0.75,
                }
            ]
            if split is not None
            else []
        )

    valid = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        verb = _clean_text(item.get("verb"))
        obj_value = item.get("direct_object")
        if isinstance(obj_value, dict):
            object_name = _clean_text(obj_value.get("name"))
        else:
            object_name = _clean_text(obj_value)
        verb, object_name = _normalize_action_phrase(verb, object_name)
        if (
            not verb
            or not object_name
            or verb in NO_ACTION_TERMS
            or object_name in NO_ACTION_TERMS
            or "short action verb" in verb
            or "short object noun" in object_name
        ):
            continue
        hand = _clean_text(item.get("hand")).removesuffix(" hand")
        if hand not in VALID_HANDS:
            hand = "unknown"
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.75))))
        except (TypeError, ValueError):
            confidence = 0.75
        object_rich = {
            "name": object_name,
            "category": "unknown",
            "specific_type": object_name,
            "visible_text": "none",
        }
        valid.append(
            {
                "verb": verb,
                "direct_object": object_name,
                "direct_object_rich": object_rich,
                "object_attributes": {
                    "color": ["unknown"],
                    "material": ["unknown"],
                    "shape": "unknown",
                    "size": "unknown",
                    "state": ["unknown"],
                    "amount": "unknown",
                    "condition": ["unknown"],
                    "part": "unknown",
                },
                "spatial_context": {
                    "location": "unknown",
                    "relative_position": "unknown",
                    "contained_in": "none",
                    "near": [],
                },
                "interaction": {
                    "hand": hand,
                    "tool_used": "none",
                    "contact_type": "unknown",
                    "role": "primary_manipulation",
                    "object_change": verb,
                    "motion_direction": "unknown",
                    "why_this_verb": "LLaVAction short action recognition output",
                },
                "hand": hand,
                "with": "none",
                "purpose": "recognized visible action",
                "manner": "unknown",
                "time_hint": "0.0-1.0",
                "confidence": round(confidence, 3),
            }
        )
    return valid, parse_ok


def _legacy_parsed(actions: list[dict]) -> dict:
    if not actions:
        return {
            "verb": "none",
            "direct_object": "none",
            "hand": "unknown",
            "with": "none",
            "purpose": "no visible hand-object action in target interval",
            "manner": "none",
            "actions": [],
        }
    primary = actions[0]
    return {
        "verb": primary["verb"],
        "direct_object": primary["direct_object"],
        "hand": primary["interaction"]["hand"],
        "with": "none",
        "purpose": "recognized visible action",
        "manner": "unknown",
        "actions": actions,
    }


def _read_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _refresh_parsed_fields(record: dict) -> dict:
    """Reapply the deterministic parser without rerunning the model."""
    actions, parse_ok = parse_actions(str(record.get("raw_output", "")))
    record["parsed"] = _legacy_parsed(actions)
    record["parsed_actions"] = actions
    record["parsed_raw_actions"] = actions
    record["valid_actions"] = actions
    record["rejected_actions"] = []
    record["n_parsed_actions"] = len(actions)
    record["n_raw_actions"] = len(actions)
    record["n_valid_actions"] = len(actions)
    record["n_rejected_actions"] = 0
    record["action_present"] = bool(actions)
    record["parse_ok"] = parse_ok
    return record


def _rewrite_records(path: Path) -> list[dict]:
    records = [_refresh_parsed_fields(record) for record in _read_completed(path)]
    records.sort(key=lambda record: int(record["window_idx"]))
    write_jsonl(records, path)
    return records


def build(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    w_path = output_dir / "w_outputs.jsonl"
    log_path = output_dir / "run.log"

    if args.overwrite and w_path.exists():
        w_path.unlink()
    existing = _read_completed(w_path)
    completed_idxs = {int(record["window_idx"]) for record in existing}

    gt_actions = load_video_actions(args.video_id, args.annotations_path)
    write_jsonl(gt_actions, output_dir / "gt.jsonl")
    alias_table = build_alias_table(Path(args.hdepic_vocab_path))
    (output_dir / "alias_table.json").write_text(
        json.dumps(alias_table, indent=2, sort_keys=True), encoding="utf-8"
    )

    config = {
        "video_id": args.video_id,
        "video_path": args.video_path,
        "model_id": MODEL_ID,
        "model_path": args.model_path,
        "model_revision": MODEL_REVISION,
        "vision_tower_path": args.vision_tower_path,
        "vision_tower_revision": VISION_TOWER_REVISION,
        "prompt_id": PROMPT_ID,
        "win_sec": args.win_sec,
        "stride_sec": args.stride_sec,
        "target_fps": args.target_fps,
        "pre_context_sec": args.pre_context_sec,
        "post_context_sec": 0.0,
        "context_fps": args.context_fps,
        "context_mode": "past",
        "overlay_frame_labels": True,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "prompt.txt").write_text(PROMPT.rstrip() + "\n", encoding="utf-8")

    windows = [
        window
        for window in make_windows(args.video_path, args.win_sec, args.stride_sec)
        if window[1] + 1e-9 >= args.start_sec and window[2] <= args.end_sec + 1e-9
    ]
    if args.window_indices:
        selected = {int(value) for value in args.window_indices.split(",")}
        windows = [window for window in windows if window[0] in selected]
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    pending = [window for window in windows if window[0] not in completed_idxs]

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"[{datetime.now(timezone.utc).isoformat()}] selected={len(windows)} "
            f"completed={len(completed_idxs)} pending={len(pending)}\n"
        )
    if not pending:
        _rewrite_records(w_path)
        print(f"All {len(windows)} selected windows are already complete: {w_path}")
        return 0

    print(f"Loading {MODEL_ID} from {args.model_path}", flush=True)
    vlm = LLaVActionVLM(
        args.model_path,
        vision_tower_path=args.vision_tower_path,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model loaded; processing {len(pending)} W windows", flush=True)

    with w_path.open("a", encoding="utf-8") as output:
        for position, (window_idx, t_start, t_end) in enumerate(pending, 1):
            win_dir = frames_root / f"win_{window_idx:05d}"
            frame_info = extract_context_frame_groups(
                args.video_path,
                window_idx,
                t_start,
                t_end,
                win_dir,
                args.target_fps,
                args.pre_context_sec,
                0.0,
                args.context_fps,
                "past",
                True,
            )
            start = time.perf_counter()
            raw_output = vlm.infer_context(
                frame_info["before_frame_paths"],
                frame_info["target_frame_paths"],
                frame_info["after_frame_paths"],
                PROMPT,
            )
            elapsed = time.perf_counter() - start
            valid_actions, parse_ok = parse_actions(raw_output)
            parsed = _legacy_parsed(valid_actions)
            record = {
                "window_idx": window_idx,
                "t_start": t_start,
                "t_end": t_end,
                "frames_path": str(win_dir),
                "frame_count": sum(
                    frame_info[key]
                    for key in (
                        "before_context_frame_count",
                        "target_frame_count",
                        "after_context_frame_count",
                    )
                ),
                "prompt_id": PROMPT_ID,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "context_mode": "past",
                "pre_context_sec": args.pre_context_sec,
                "post_context_sec": 0.0,
                "context_fps": args.context_fps,
                "overlay_frame_labels": True,
                "before_context_frames_path": frame_info["before_frames_path"],
                "target_frames_path": frame_info["target_frames_path"],
                "after_context_frames_path": frame_info["after_frames_path"],
                "before_context_frame_count": frame_info["before_context_frame_count"],
                "target_frame_count": frame_info["target_frame_count"],
                "after_context_frame_count": frame_info["after_context_frame_count"],
                "raw_output": raw_output,
                "parsed": parsed,
                "parsed_actions": valid_actions,
                "parsed_raw_actions": valid_actions,
                "valid_actions": valid_actions,
                "rejected_actions": [],
                "n_parsed_actions": len(valid_actions),
                "n_raw_actions": len(valid_actions),
                "n_valid_actions": len(valid_actions),
                "n_rejected_actions": 0,
                "action_present": bool(valid_actions),
                "parse_ok": parse_ok,
                "inference_time_sec": elapsed,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            line = (
                f"[{position}/{len(pending)}] window={window_idx} "
                f"parse_ok={parse_ok} actions={len(valid_actions)} "
                f"time={elapsed:.2f}s raw={raw_output!r}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(line + "\n")

    records = _rewrite_records(w_path)
    print(f"W complete: {w_path} ({len(records)} records)")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower-path", required=True)
    parser.add_argument("--annotations-path", required=True)
    parser.add_argument("--hdepic-vocab-path", required=True)
    parser.add_argument("--start-sec", type=float, default=200.0)
    parser.add_argument("--end-sec", type=float, default=290.0)
    parser.add_argument("--win-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--pre-context-sec", type=float, default=3.0)
    parser.add_argument("--context-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--window-indices", default="")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
