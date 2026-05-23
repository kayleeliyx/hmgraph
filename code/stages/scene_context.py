"""Rebuild scene rolling context v1.1 and combined W/scene evidence.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.run_hmgraph_pipeline --video-path /path/to/video.mp4 --output-dir /path/to/output --start-sec START --end-sec END --range-label START_END --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from code.scene.context_builder import (
        calibrate_action_presence,
        extract_stable_objects_from_scene,
        extract_stable_objects_from_w_action,
        render_scene_context_object_only_v1_1,
        update_scene_rolling_context_v1_1,
    )
    from code.scene.object_utils import canonicalize_object_name, is_generic_object_name
    from code.scene.observation import summarize_scene_observation
except ImportError:
    from code.scene.context_builder import (
        calibrate_action_presence,
        extract_stable_objects_from_scene,
        extract_stable_objects_from_w_action,
        render_scene_context_object_only_v1_1,
        update_scene_rolling_context_v1_1,
    )
    from code.scene.object_utils import canonicalize_object_name, is_generic_object_name
    from code.scene.observation import summarize_scene_observation


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[dict], path: str | Path, overwrite: bool) -> None:
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _assert_exact_range(records: list[dict], start_sec: float, end_sec: float, label: str) -> None:
    expected = [(idx, float(idx), float(idx + 1)) for idx in range(int(start_sec), int(end_sec))]
    actual = [(int(r["window_idx"]), float(r["t_start"]), float(r["t_end"])) for r in records]
    if actual != expected:
        raise RuntimeError(f"Expected {label} windows {expected}; got {actual}")


def _valid_actions(record: dict) -> list[dict]:
    actions = record.get("valid_actions")
    if isinstance(actions, list):
        return [a for a in actions if isinstance(a, dict)]
    parsed = record.get("parsed") or {}
    actions = parsed.get("actions")
    if isinstance(actions, list):
        return [a for a in actions if isinstance(a, dict)]
    return []


def _valid_action_summary(record: dict) -> list[dict]:
    return [
        {
            "verb": action.get("verb", ""),
            "object": action.get("direct_object", ""),
            "confidence": float(action.get("confidence", 0.0)),
        }
        for action in _valid_actions(record)
    ]


def _w_action_summary(record: dict) -> dict:
    return {
        "action_present": bool(record.get("action_present")),
        "valid_actions": _valid_action_summary(record),
        "raw_action_candidate_present": bool(record.get("parsed_raw_actions")),
        "parse_ok": bool(record.get("parse_ok")),
    }


def _pairs(record: dict) -> str:
    actions = _valid_actions(record)
    if not actions:
        return "none"
    return ", ".join(f"{a.get('verb', '')}/{a.get('direct_object', '')}" for a in actions)


def _status_counts(statuses: Iterable[str]) -> dict[str, int]:
    counts = {"visible_action": 0, "uncertain_action": 0, "no_action": 0}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    return counts


def _object_names(objects: list[dict]) -> list[str]:
    names = []
    for obj in objects:
        name = obj.get("raw_name") or obj.get("canonical_name") or ""
        if name:
            names.append(str(name))
    return names


def _target_checks(stable_objects: list[dict]) -> dict:
    raw_names = {canonicalize_object_name(obj.get("raw_name", "")) for obj in stable_objects}
    canonical_names = {canonicalize_object_name(obj.get("canonical_name", "")) for obj in stable_objects}
    all_names = raw_names | canonical_names
    return {
        "coffee_machine": "coffee machine" in all_names,
        "mug": "mug" in all_names,
        "button": "button" in all_names,
        "refrigerator": "refrigerator" in all_names,
        "cupboard": "cupboard" in all_names,
    }


def _duplicate_alias_groups(stable_objects: list[dict]) -> list[list[str]]:
    groups: dict[str, set[str]] = {}
    for obj in stable_objects:
        canonical = canonicalize_object_name(obj.get("canonical_name", ""))
        raw = canonicalize_object_name(obj.get("raw_name", ""))
        groups.setdefault(canonical, set()).add(raw)
    return [sorted(values) for values in groups.values() if len(values) > 1]


def _quality_report(w_records: list[dict], scene_records: list[dict], traces: list[dict]) -> dict:
    raw_statuses = []
    calibrated_statuses = []
    raw_mismatch = 0
    calibrated_mismatch = 0
    no_action_useful = 0
    downgraded = []
    for w_record, scene_record, trace in zip(w_records, scene_records, traces):
        scene = scene_record.get("scene_observation") or {}
        presence = scene.get("action_presence") or {}
        raw_status = presence.get("status", "uncertain_action")
        calibrated = trace["calibrated_action_presence"]
        raw_statuses.append(raw_status)
        calibrated_statuses.append(calibrated["calibrated_status"])
        w_present = bool(w_record.get("action_present"))
        if w_present != (raw_status == "visible_action"):
            raw_mismatch += 1
        if w_present != (calibrated["calibrated_status"] == "visible_action"):
            calibrated_mismatch += 1
        if raw_status == "no_action" and bool(scene.get("visible_objects")):
            no_action_useful += 1
        if raw_status == "visible_action" and calibrated["calibrated_status"] == "uncertain_action":
            downgraded.append(f"{float(w_record['t_start']):.1f}-{float(w_record['t_end']):.1f}: {calibrated['reason']}")
    final_context = traces[-1]["scene_context_after_window"] if traces else {}
    stable_objects = final_context.get("stable_objects") or []
    generic_count = sum(
        1 for obj in stable_objects if is_generic_object_name(obj.get("canonical_name") or obj.get("raw_name", ""))
    )
    return {
        "windows_processed": len(w_records),
        "scene_parse_failures": sum(1 for record in scene_records if not record.get("parse_ok")),
        "raw_scene_action_presence_counts": _status_counts(raw_statuses),
        "calibrated_action_presence_counts": _status_counts(calibrated_statuses),
        "W_scene_action_present_mismatch_count_raw": raw_mismatch,
        "W_scene_action_present_mismatch_count_calibrated": calibrated_mismatch,
        "no_action_windows_with_visible_objects": no_action_useful,
        "stable_objects_by_20s": [
            f"{obj.get('raw_name')}|canonical={obj.get('canonical_name')}|part_of={obj.get('part_of')}@{obj.get('location')}"
            for obj in stable_objects
        ],
        "target_object_checks": _target_checks(stable_objects),
        "generic_object_count": generic_count,
        "duplicate_alias_groups_detected": _duplicate_alias_groups(stable_objects),
        "windows_downgraded_to_uncertain_action": downgraded,
        "notes": [
            "scene_obs_v1 used as scene source",
            "W_action valid actions used as action/object evidence",
            "scene_obs_v2 not used",
            "no model inference run",
            "object normalization is conservative and keeps raw_name/canonical_name/part_of",
        ],
    }


def _print_report(w_records: list[dict], scene_records: list[dict], traces: list[dict], report: dict) -> None:
    print(
        "interval | W valid verb/object | W action_present | raw scene status/confidence | "
        "calibrated status/confidence | scene_type | main objects | objects added from W_action | "
        "current_focus | object_only_context_text_for_future_w"
    )
    for w_record, scene_record, trace in zip(w_records, scene_records, traces):
        scene = scene_record["scene_observation"]
        summary = summarize_scene_observation(scene)
        presence = scene["action_presence"]
        calibrated = trace["calibrated_action_presence"]
        print(
            f"{float(w_record['t_start']):.1f}-{float(w_record['t_end']):.1f} | "
            f"{_pairs(w_record)} | {bool(w_record.get('action_present'))} | "
            f"{presence['status']}/{float(presence['confidence']):.2f} | "
            f"{calibrated['calibrated_status']}/{float(calibrated['confidence']):.2f} | "
            f"{scene['scene_type']} | {', '.join(summary['main_objects']) or 'none'} | "
            f"{', '.join(trace['objects_added_from_W_action']) or 'none'} | "
            f"{trace['scene_context_after_window']['current_focus']} | "
            f"{' '.join(trace['object_only_context_text_for_future_w'].split())}"
        )
    print("stable_objects_by_20s=" + "; ".join(report["stable_objects_by_20s"]))
    print("target_object_checks=" + json.dumps(report["target_object_checks"], sort_keys=True))
    print("windows_downgraded_to_uncertain_action=" + ("; ".join(report["windows_downgraded_to_uncertain_action"]) or "none"))
    print("raw_scene_action_presence_counts=" + json.dumps(report["raw_scene_action_presence_counts"], sort_keys=True))
    print("calibrated_action_presence_counts=" + json.dumps(report["calibrated_action_presence_counts"], sort_keys=True))
    print(f"generic_object_count={report['generic_object_count']}")
    print("duplicate_alias_groups_detected=" + json.dumps(report["duplicate_alias_groups_detected"], sort_keys=True))
    print(f"no_action_windows_with_visible_objects={report['no_action_windows_with_visible_objects']}")


def run(args: argparse.Namespace) -> dict:
    w_records = [
        record
        for record in read_jsonl(args.w_action_jsonl)
        if float(record["t_start"]) >= args.start_sec - 1e-9 and float(record["t_end"]) <= args.end_sec + 1e-9
    ]
    scene_records = [
        record
        for record in read_jsonl(args.scene_jsonl)
        if float(record["t_start"]) >= args.start_sec - 1e-9 and float(record["t_end"]) <= args.end_sec + 1e-9
    ]
    _assert_exact_range(w_records, args.start_sec, args.end_sec, "W_action")
    _assert_exact_range(scene_records, args.start_sec, args.end_sec, "scene_obs_v1")
    scene_by_idx = {int(record["window_idx"]): record for record in scene_records}

    rolling_context = None
    traces = []
    combined = []
    ordered_scene_records = []
    for w_record in w_records:
        scene_record = scene_by_idx[int(w_record["window_idx"])]
        ordered_scene_records.append(scene_record)
        context_before = rolling_context
        calibrated = calibrate_action_presence(scene_record, w_record)
        scene_objects = extract_stable_objects_from_scene(scene_record)
        w_objects = extract_stable_objects_from_w_action(w_record)
        rolling_context = update_scene_rolling_context_v1_1(rolling_context, scene_record, w_record, float(w_record["t_end"]))
        context_text = render_scene_context_object_only_v1_1(rolling_context)
        scene_summary = summarize_scene_observation(scene_record["scene_observation"])
        w_summary = _w_action_summary(w_record)
        trace = {
            "window_idx": int(w_record["window_idx"]),
            "t_start": float(w_record["t_start"]),
            "t_end": float(w_record["t_end"]),
            "scene_context_before_window": context_before,
            "raw_scene_action_presence": (scene_record.get("scene_observation") or {}).get("action_presence", {}),
            "calibrated_action_presence": calibrated,
            "scene_observation_summary": scene_summary,
            "w_action_summary": w_summary,
            "objects_added_from_scene": _object_names(scene_objects),
            "objects_added_from_W_action": _object_names(w_objects),
            "scene_context_after_window": rolling_context,
            "object_only_context_text_for_future_w": context_text,
        }
        traces.append(trace)
        combined.append(
            {
                "window_idx": int(w_record["window_idx"]),
                "t_start": float(w_record["t_start"]),
                "t_end": float(w_record["t_end"]),
                "W_action": {
                    "source_file": Path(args.w_action_jsonl).name,
                    "record": w_record,
                    "summary": w_summary,
                },
                "scene_observation": {
                    "source_file": Path(args.scene_jsonl).name,
                    "record": scene_record,
                    "summary": scene_summary,
                    "calibrated_action_presence": calibrated,
                },
                "rolling_context_v1_1": {
                    "before_window": context_before,
                    "after_window": rolling_context,
                    "objects_added_from_scene": trace["objects_added_from_scene"],
                    "objects_added_from_W_action": trace["objects_added_from_W_action"],
                    "object_only_context_text_for_future_w": context_text,
                },
            }
        )

    write_jsonl(traces, args.rolling_output_jsonl, args.overwrite)
    write_jsonl(combined, args.combined_output_jsonl, args.overwrite)
    report = _quality_report(w_records, ordered_scene_records, traces)
    report_path = Path(args.quality_report_json)
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_report(w_records, ordered_scene_records, traces, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-action-jsonl", required=True)
    parser.add_argument("--scene-jsonl", required=True)
    parser.add_argument("--rolling-output-jsonl", required=True)
    parser.add_argument("--combined-output-jsonl", required=True)
    parser.add_argument("--quality-report-json", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
