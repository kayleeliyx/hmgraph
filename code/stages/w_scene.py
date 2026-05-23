"""Run scene_obs_v1 and combine W action + scene observations.

# Run from /data2/yaxuanli/hmgraph_codex:
# python -m code.run_hmgraph_pipeline --video-path /path/to/video.mp4 --output-dir /path/to/output --start-sec START --end-sec END --range-label START_END --overwrite
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from code.utils import paths
from code.w.build import build_w
from code.utils.gemma_vlm import GemmaVLM
from code.scene.observation import (
    derive_action_present_from_scene,
    parse_scene_observation_json_with_repair,
    summarize_scene_observation,
)
from code.scene.observation_prompt import SCENE_OBS_PROMPT, SCENE_OBS_PROMPT_ID
from code.scene.rolling_context import render_scene_context_object_only, update_scene_rolling_context
from code.w.window_extractor import extract_context_frame_groups


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


def _assert_exact_range(records: list[dict], start_sec: float, end_sec: float) -> None:
    expected = [(idx, float(idx), float(idx + 1)) for idx in range(int(start_sec), int(end_sec))]
    actual = [(int(r["window_idx"]), float(r["t_start"]), float(r["t_end"])) for r in records]
    if actual != expected:
        raise RuntimeError(f"Expected W windows {expected}; got {actual}")


def _ensure_w_action(args: argparse.Namespace) -> tuple[list[dict], str]:
    w_path = Path(args.w_action_jsonl)
    if w_path.exists() and not args.overwrite_w_action:
        return read_jsonl(w_path), "reused"
    if w_path.exists() and not args.overwrite_w_action:
        raise FileExistsError(f"Refusing to overwrite W action file without --overwrite-w-action: {w_path}")
    build_args = argparse.Namespace(
        video_path=args.video_path,
        video_id=paths.DEFAULT_VIDEO_ID,
        output_dir=str(w_path.parent),
        frames_root=None,
        w_output=None,
        output_jsonl=str(w_path),
        annotations_path=str(paths.HD_EPIC_NARRATIONS),
        hdepic_vocab_path=str(paths.HD_EPIC_VERB_CLASSES.parent),
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        win_sec=1.0,
        stride_sec=1.0,
        target_fps=args.target_fps,
        n_frames_per_window=4,
        pre_context_sec=args.pre_context_sec,
        post_context_sec=args.post_context_sec,
        context_fps=args.context_fps,
        context_mode=args.context_mode,
        overlay_frame_labels=True,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        max_windows=None,
        overwrite=True,
        run_log=None,
    )
    return build_w(build_args), "generated"


def _valid_action_summary(record: dict) -> list[dict]:
    return [
        {
            "verb": action.get("verb", ""),
            "object": action.get("direct_object", ""),
            "confidence": float(action.get("confidence", 0.0)),
        }
        for action in record.get("valid_actions") or []
    ]


def _w_action_summary(record: dict) -> dict:
    return {
        "action_present": bool(record.get("action_present")),
        "valid_actions": _valid_action_summary(record),
        "raw_action_candidate_present": bool(record.get("parsed_raw_actions")),
        "parse_ok": bool(record.get("parse_ok")),
    }


def _pairs(record: dict) -> str:
    actions = record.get("valid_actions") or []
    if not actions:
        return "none"
    return ", ".join(f"{a.get('verb', '')}/{a.get('direct_object', '')}" for a in actions)


def _status_counts(records: list[dict]) -> dict[str, int]:
    counts = {"visible_action": 0, "uncertain_action": 0, "no_action": 0}
    for record in records:
        status = (record.get("scene_observation") or {}).get("action_presence", {}).get("status", "uncertain_action")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _mismatch_reason(w_record: dict, scene_record: dict) -> str:
    w_present = bool(w_record.get("action_present"))
    scene = scene_record.get("scene_observation") or {}
    presence = scene.get("action_presence") or {}
    scene_present = derive_action_present_from_scene(scene)
    if w_present and not scene_present:
        return f"W has valid action; scene status={presence.get('status')} evidence={presence.get('evidence')}"
    if scene_present and not w_present:
        return "scene sees visible action but W validity gate has no valid action"
    return "matched"


def _object_names_from_context(context: dict) -> set[str]:
    return {str(obj.get("name", "")).lower() for obj in context.get("stable_objects") or []}


def _print_report(
    w_records: list[dict],
    scene_records: list[dict],
    context_traces: list[dict],
    w_action_status: str,
) -> None:
    scene_by_idx = {int(r["window_idx"]): r for r in scene_records}
    trace_by_idx = {int(r["window_idx"]): r for r in context_traces}
    print(
        "interval | W action valid verb/object | W action_present | scene status | "
        "scene confidence | scene_type | view_context | main visible objects | "
        "current_focus | object_only_context_text_for_future_w"
    )
    for w_record in w_records:
        scene_record = scene_by_idx[int(w_record["window_idx"])]
        trace = trace_by_idx[int(w_record["window_idx"])]
        scene = scene_record["scene_observation"]
        summary = summarize_scene_observation(scene)
        action_presence = scene["action_presence"]
        print(
            f"{float(w_record['t_start']):.1f}-{float(w_record['t_end']):.1f} | "
            f"{_pairs(w_record)} | {bool(w_record.get('action_present'))} | "
            f"{action_presence['status']} | {action_presence['confidence']:.2f} | "
            f"{scene['scene_type']} | {scene['view_context']} | "
            f"{', '.join(summary['main_objects']) or 'none'} | "
            f"{scene['current_focus']} | "
            f"{' '.join(trace['object_only_context_text_for_future_w'].split())}"
        )

    counts = _status_counts(scene_records)
    mismatches = [
        (w_record, scene_by_idx[int(w_record["window_idx"])])
        for w_record in w_records
        if bool(w_record.get("action_present"))
        != derive_action_present_from_scene(scene_by_idx[int(w_record["window_idx"])]["scene_observation"])
    ]
    final_context = context_traces[-1]["scene_context_after_window"] if context_traces else {}
    stable_names = _object_names_from_context(final_context)
    target_names = ["coffee machine", "mug", "button", "refrigerator", "cupboard"]
    no_action_useful = sum(
        1
        for record in scene_records
        if (record["scene_observation"]["action_presence"]["status"] == "no_action")
        and bool(record["scene_observation"].get("visible_objects"))
    )
    print(f"windows_processed={len(scene_records)}")
    print(f"W_action_file_status={w_action_status}")
    print(f"scene_parse_failures={sum(1 for record in scene_records if not record.get('parse_ok'))}")
    print(
        "scene_action_presence_counts="
        f"visible_action:{counts.get('visible_action', 0)} "
        f"uncertain_action:{counts.get('uncertain_action', 0)} "
        f"no_action:{counts.get('no_action', 0)}"
    )
    print(f"W_scene_action_present_mismatch_count={len(mismatches)}")
    print(
        "mismatch_windows="
        + (
            "; ".join(
                f"{float(w['t_start']):.1f}-{float(w['t_end']):.1f}: {_mismatch_reason(w, s)}"
                for w, s in mismatches
            )
            if mismatches
            else "none"
        )
    )
    print(
        "stable_objects_at_20s="
        + "; ".join(
            f"{obj['name']}@{obj['location']}({obj['state']})"
            for obj in final_context.get("stable_objects", [])
        )
    )
    print(
        "target_objects_observed="
        + ", ".join(f"{name}:{name in stable_names}" for name in target_names)
    )
    print(f"no_action_windows_with_visible_objects={no_action_useful}")


def run(args: argparse.Namespace) -> list[dict]:
    if args.context_mode != "past" or args.post_context_sec != 0:
        raise RuntimeError("scene_obs_v1 diagnostic requires context_mode=past and post_context_sec=0")
    w_records, w_action_status = _ensure_w_action(args)
    w_records = [
        record
        for record in w_records
        if float(record["t_start"]) >= args.start_sec - 1e-9
        and float(record["t_end"]) <= args.end_sec + 1e-9
    ]
    _assert_exact_range(w_records, args.start_sec, args.end_sec)

    scene_output = Path(args.scene_output_jsonl)
    rolling_output = Path(args.rolling_context_jsonl)
    combined_output = Path(args.combined_output_jsonl)
    frame_root = Path(args.scene_frame_output_dir)
    for out in (scene_output, rolling_output, combined_output):
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {out}")
    if frame_root.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing diagnostic frames without --overwrite: {frame_root}")

    vlm = GemmaVLM(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.scene_max_new_tokens,
    )
    scene_records: list[dict] = []
    context_traces: list[dict] = []
    combined_records: list[dict] = []
    rolling_context: dict | None = None

    for w_record in w_records:
        frame_info = extract_context_frame_groups(
            args.video_path,
            int(w_record["window_idx"]),
            float(w_record["t_start"]),
            float(w_record["t_end"]),
            frame_root / f"win_{int(w_record['window_idx']):05d}",
            args.target_fps,
            args.pre_context_sec,
            args.post_context_sec,
            args.context_fps,
            args.context_mode,
            True,
        )
        if frame_info["after_context_frame_count"] != 0:
            raise RuntimeError(f"W window {w_record['window_idx']} has after-context frames")
        start = time.perf_counter()
        raw_scene_output = vlm.infer_context(
            frame_info["before_frame_paths"],
            frame_info["target_frame_paths"],
            frame_info["after_frame_paths"],
            SCENE_OBS_PROMPT,
        )
        inference_time = time.perf_counter() - start
        scene_observation, parse_ok, repair_info = parse_scene_observation_json_with_repair(raw_scene_output)
        scene_record = {
            "window_idx": int(w_record["window_idx"]),
            "t_start": float(w_record["t_start"]),
            "t_end": float(w_record["t_end"]),
            "prompt_id": SCENE_OBS_PROMPT_ID,
            "model_id": args.model_id,
            "target_frame_count": frame_info["target_frame_count"],
            "before_context_frame_count": frame_info["before_context_frame_count"],
            "after_context_frame_count": frame_info["after_context_frame_count"],
            "raw_scene_output": raw_scene_output,
            "scene_observation": scene_observation,
            "parse_ok": parse_ok,
            "parse_repair_info": repair_info,
            "scene_action_present": derive_action_present_from_scene(scene_observation),
            "inference_time_sec": inference_time,
        }
        context_before = rolling_context
        rolling_context = update_scene_rolling_context(
            rolling_context,
            scene_record,
            w_record,
            float(w_record["t_end"]),
        )
        context_text = render_scene_context_object_only(rolling_context)
        scene_summary = summarize_scene_observation(scene_observation)
        w_summary = _w_action_summary(w_record)
        trace = {
            "window_idx": int(w_record["window_idx"]),
            "t_start": float(w_record["t_start"]),
            "t_end": float(w_record["t_end"]),
            "scene_context_before_window": context_before,
            "scene_observation_summary": scene_summary,
            "w_action_summary": w_summary,
            "scene_context_after_window": rolling_context,
            "object_only_context_text_for_future_w": context_text,
        }
        combined = {
            "window_idx": int(w_record["window_idx"]),
            "t_start": float(w_record["t_start"]),
            "t_end": float(w_record["t_end"]),
            "W_action": {
                "source_file": Path(args.w_action_jsonl).name,
                "record": w_record,
                "summary": w_summary,
            },
            "scene_observation": {
                "source_file": Path(args.scene_output_jsonl).name,
                "record": scene_record,
                "summary": scene_summary,
            },
            "rolling_context": {
                "before_window": context_before,
                "after_window": rolling_context,
                "object_only_context_text_for_future_w": context_text,
            },
        }
        scene_records.append(scene_record)
        context_traces.append(trace)
        combined_records.append(combined)

    write_jsonl(scene_records, scene_output, args.overwrite)
    write_jsonl(context_traces, rolling_output, args.overwrite)
    write_jsonl(combined_records, combined_output, args.overwrite)
    _print_report(w_records, scene_records, context_traces, w_action_status)
    return scene_records


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--w-action-jsonl", required=True)
    parser.add_argument("--scene-output-jsonl", required=True)
    parser.add_argument("--scene-frame-output-dir", required=True)
    parser.add_argument("--rolling-context-jsonl", required=True)
    parser.add_argument("--combined-output-jsonl", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--target-fps", type=float, default=8)
    parser.add_argument("--pre-context-sec", type=float, default=3)
    parser.add_argument("--post-context-sec", type=float, default=0)
    parser.add_argument("--context-fps", type=float, default=2)
    parser.add_argument("--context-mode", choices=("past", "none", "both"), default="past")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-w-action", action="store_true")
    parser.add_argument("--model-id", default=paths.DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--scene-max-new-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
