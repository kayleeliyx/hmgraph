"""Build raw W records with temporal context frames and Gemma VLM.

Inputs:
  Video path, output directory/jsonl, model id/path, target window settings,
  and before/after context frame settings.
Outputs:
  frames/win_NNNNN/{before,target,after}/*.jpg, W jsonl records, gt.jsonl,
  alias_table.json, config.json, and run.log.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from code.utils import paths
from code.utils.alias_table import build as build_alias_table
from code.utils.gemma_vlm import GemmaVLM
from code.utils.hdepic_loader import load_video_actions, write_jsonl
from code.w.prompt import PROMPT, PROMPT_ID, filter_valid_target_actions, parse
from code.w.window_extractor import extract_context_frame_groups, make_windows, video_duration_sec


def _git_sha(repo: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _append_log(run_log: Path, message: str) -> None:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _assert_output_allowed(paths_to_check: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths_to_check if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output files without --overwrite: "
            + ", ".join(existing)
        )


def _config_output_path(output_dir: Path, args: argparse.Namespace) -> Path:
    """Return config path, preserving config.json for default full-run behavior."""
    if args.start_sec == 0.0 and args.end_sec is None:
        return output_dir / "config.json"
    start = f"{args.start_sec:g}".replace(".", "p")
    end = "end" if args.end_sec is None else f"{args.end_sec:g}".replace(".", "p")
    fps = f"{args.target_fps:g}".replace(".", "p")
    return output_dir / f"config_{start}_{end}_dense{fps}.json"


def _filter_windows_by_range(
    windows: list[tuple[int, float, float]],
    start_sec: float,
    end_sec: float | None,
) -> list[tuple[int, float, float]]:
    """Keep W windows fully contained in [start_sec, end_sec]."""
    filtered = []
    for window_idx, t_start, t_end in windows:
        if t_start + 1e-9 < start_sec:
            continue
        if end_sec is not None and t_end > end_sec + 1e-9:
            continue
        filtered.append((window_idx, t_start, t_end))
    return filtered


def _flatten_parsed_action(actions: list[dict]) -> tuple[dict, bool]:
    """Return legacy parsed fields plus action_present for existing L1 code."""
    if not actions:
        return (
            {
                "verb": "none",
                "direct_object": "none",
                "hand": "unknown",
                "with": "none",
                "purpose": "no visible hand-object action in target interval",
                "manner": "none",
                "actions": [],
            },
            False,
        )
    primary = next(
        (
            action
            for action in actions
            if action.get("interaction", {}).get("role") == "primary_manipulation"
        ),
        actions[0],
    )
    return (
        {
            "verb": primary.get("verb", ""),
            "direct_object": primary.get("direct_object", ""),
            "hand": primary.get("hand", "unknown"),
            "with": primary.get("with", "none"),
            "purpose": primary.get("interaction", {}).get("why_this_verb", primary.get("purpose", "unknown")),
            "manner": primary.get("interaction", {}).get(
                "motion_direction", primary.get("manner", "")
            ),
            "actions": actions,
        },
        True,
    )


def build_w(args: argparse.Namespace) -> list[dict]:
    """Produce W records for the requested video and write sidecar files."""
    output_dir = Path(args.output_dir)
    frames_root = Path(args.frames_root) if args.frames_root else output_dir / "frames"
    run_log = Path(args.run_log) if args.run_log else output_dir / "run.log"
    w_output = (
        Path(args.output_jsonl or args.w_output)
        if (args.output_jsonl or args.w_output)
        else output_dir / "w_outputs.jsonl"
    )
    config_path = _config_output_path(output_dir, args)
    _assert_output_allowed(
        [
            w_output,
            output_dir / "alias_table.json",
            output_dir / "gt.jsonl",
            config_path,
        ],
        args.overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    alias_table = build_alias_table(Path(args.hdepic_vocab_path))
    (output_dir / "alias_table.json").write_text(
        json.dumps(alias_table, indent=2, sort_keys=True), encoding="utf-8"
    )

    gt_actions = load_video_actions(args.video_id, args.annotations_path)
    write_jsonl(gt_actions, output_dir / "gt.jsonl")

    config = {
        "video_id": args.video_id,
        "video_path": args.video_path,
        "model_id": args.model_id,
        "prompt_id": PROMPT_ID,
        "win_sec": args.win_sec,
        "stride_sec": args.stride_sec,
        "target_fps": args.target_fps,
        "n_frames_per_window": args.n_frames_per_window,
        "pre_context_sec": args.pre_context_sec,
        "post_context_sec": args.post_context_sec,
        "context_fps": args.context_fps,
        "context_mode": args.context_mode,
        "overlay_frame_labels": args.overlay_frame_labels,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    windows = make_windows(args.video_path, args.win_sec, args.stride_sec)
    windows = _filter_windows_by_range(windows, args.start_sec, args.end_sec)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if args.start_sec == 10.0 and args.end_sec == 20.0:
        expected = [(idx, float(idx), float(idx + 1)) for idx in range(10, 20)]
        actual = [(idx, float(start), float(end)) for idx, start, end in windows]
        if actual != expected:
            raise RuntimeError(
                "Expected exactly W windows 10.0-11.0 through 19.0-20.0; "
                f"got {actual}"
            )
    video_duration = video_duration_sec(args.video_path)

    _append_log(
        run_log,
        (
            f"[build_w] W windows={len(windows)} model_id={args.model_id} "
            f"prompt_id={PROMPT_ID} context_mode={args.context_mode}"
        ),
    )
    vlm = GemmaVLM(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
    )

    records = []
    with w_output.open("w", encoding="utf-8") as f:
        for window_idx, t_start, t_end in windows:
            win_dir = frames_root / f"win_{window_idx:05d}"
            frame_info = extract_context_frame_groups(
                args.video_path,
                window_idx,
                t_start,
                t_end,
                win_dir,
                args.target_fps,
                args.pre_context_sec,
                args.post_context_sec,
                args.context_fps,
                args.context_mode,
                args.overlay_frame_labels,
            )
            for key in ("before_frames_path", "target_frames_path", "after_frames_path"):
                assert Path(frame_info[key]).exists(), f"Missing frame path: {frame_info[key]}"
            if args.context_mode == "past":
                assert frame_info["after_context_frame_count"] == 0, (
                    f"W window {window_idx} past context expected 0 after frames, got "
                    f"{frame_info['after_context_frame_count']}"
                )
            assert args.overlay_frame_labels is True, "overlay_frame_labels must be true for this run"
            for target_path in frame_info["target_frame_paths"]:
                assert Path(target_path).exists(), f"Missing target frame: {target_path}"
            if abs((t_end - t_start) - args.win_sec) < 1e-6 and t_end < video_duration - 1e-6:
                expected_target_frames = max(1, int(round((t_end - t_start) * args.target_fps)))
                assert frame_info["target_frame_count"] == expected_target_frames, (
                    f"W window {window_idx} expected {expected_target_frames} target frames, got "
                    f"{frame_info['target_frame_count']}"
                )
            assert PROMPT_ID == "w_atomic_v4", f"Unexpected prompt_id: {PROMPT_ID}"

            start = time.perf_counter()
            raw_output = vlm.infer_context(
                frame_info["before_frame_paths"],
                frame_info["target_frame_paths"],
                frame_info["after_frame_paths"],
                PROMPT,
            )
            inference_time = time.perf_counter() - start
            parsed_raw_actions, parse_ok = parse(raw_output, t_start, t_end)
            valid_actions, rejected_actions = filter_valid_target_actions(parsed_raw_actions)
            parsed, action_present = _flatten_parsed_action(valid_actions)
            frame_count = (
                frame_info["before_context_frame_count"]
                + frame_info["target_frame_count"]
                + frame_info["after_context_frame_count"]
            )
            record = {
                "window_idx": window_idx,
                "t_start": t_start,
                "t_end": t_end,
                "frames_path": str(win_dir),
                "frame_count": frame_count,
                "prompt_id": PROMPT_ID,
                "model_id": args.model_id,
                "context_mode": args.context_mode,
                "pre_context_sec": args.pre_context_sec,
                "post_context_sec": args.post_context_sec,
                "context_fps": args.context_fps,
                "overlay_frame_labels": args.overlay_frame_labels,
                "before_context_frames_path": frame_info["before_frames_path"],
                "target_frames_path": frame_info["target_frames_path"],
                "after_context_frames_path": frame_info["after_frames_path"],
                "before_context_frame_count": frame_info["before_context_frame_count"],
                "target_frame_count": frame_info["target_frame_count"],
                "after_context_frame_count": frame_info["after_context_frame_count"],
                "raw_output": raw_output,
                "parsed": parsed,
                "parsed_actions": valid_actions,
                "parsed_raw_actions": parsed_raw_actions,
                "valid_actions": valid_actions,
                "rejected_actions": rejected_actions,
                "n_parsed_actions": len(valid_actions),
                "n_raw_actions": len(parsed_raw_actions),
                "n_valid_actions": len(valid_actions),
                "n_rejected_actions": len(rejected_actions),
                "action_present": action_present,
                "parse_ok": parse_ok,
                "inference_time_sec": inference_time,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)
            _append_log(
                run_log,
                (
                    f"[build_w] W window_idx={window_idx} parse_ok={parse_ok} "
                    f"action_present={action_present} raw_actions={len(parsed_raw_actions)} "
                    f"valid_actions={len(valid_actions)} rejected_actions={len(rejected_actions)}"
                ),
            )
            if args.max_windows is not None and args.max_windows <= 20:
                print(
                    json.dumps(
                        {
                            "window_idx": window_idx,
                            "t_start": t_start,
                            "t_end": t_end,
                            "before_context_frame_count": frame_info["before_context_frame_count"],
                            "target_frame_count": frame_info["target_frame_count"],
                            "after_context_frame_count": frame_info["after_context_frame_count"],
                            "raw_output": raw_output,
                            "parsed": parsed,
                            "action_present": action_present,
                            "n_raw_actions": len(parsed_raw_actions),
                            "n_valid_actions": len(valid_actions),
                            "n_rejected_actions": len(rejected_actions),
                            "rejected_actions": rejected_actions,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
    return records


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default=str(paths.DEFAULT_VIDEO_PATH))
    parser.add_argument("--video-id", default=paths.DEFAULT_VIDEO_ID)
    parser.add_argument("--output-dir", default=str(paths.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--w-output", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--annotations-path", default=str(paths.HD_EPIC_NARRATIONS))
    parser.add_argument("--hdepic-vocab-path", default=str(paths.HD_EPIC_VERB_CLASSES.parent))
    parser.add_argument("--model-id", default=paths.DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--win-sec", type=float, default=1)
    parser.add_argument("--stride-sec", type=float, default=1)
    parser.add_argument("--target-fps", type=float, default=4)
    parser.add_argument("--n-frames-per-window", type=int, default=4)
    parser.add_argument("--pre-context-sec", type=float, default=3)
    parser.add_argument("--post-context-sec", type=float, default=0)
    parser.add_argument("--context-fps", type=float, default=2)
    parser.add_argument("--context-mode", choices=("none", "past", "both"), default="past")
    parser.add_argument("--overlay-frame-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-log", default=None)
    args = parser.parse_args(argv)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":")[-1]))
    records = build_w(args)
    print(f"W records written: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
