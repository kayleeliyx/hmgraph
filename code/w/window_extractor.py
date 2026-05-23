"""Create fixed W windows and sample frames for each window.

Inputs:
  Video path, window/stride seconds, output frame directory.
Outputs:
  Window tuples and JPEG frames under frames/win_NNNNN/.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2

Window = Tuple[int, float, float]


def video_duration_sec(video_path: str | Path) -> float:
    """Return video duration from ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def make_windows(
    video_path: str | Path, win_sec: float = 3, stride_sec: float = 3
) -> List[Window]:
    """Return fixed non-overlapping W windows covering the video duration."""
    duration = video_duration_sec(video_path)
    windows = []
    idx = 0
    t_start = 0.0
    while t_start < duration:
        t_end = min(t_start + win_sec, duration)
        windows.append((idx, round(t_start, 6), round(t_end, 6)))
        idx += 1
        t_start += stride_sec
    return windows


def _sample_times(t_start: float, t_end: float, n_frames: int) -> Sequence[float]:
    if n_frames <= 1:
        return [(t_start + t_end) / 2.0]
    eps = min(0.05, max((t_end - t_start) / 100.0, 0.0))
    start = t_start + eps
    end = max(start, t_end - eps)
    step = (end - start) / float(n_frames - 1)
    return [start + i * step for i in range(n_frames)]


def _sample_times_at_fps(t_start: float, t_end: float, fps: float) -> Sequence[float]:
    """Return uniformly spaced sample times for [t_start, t_end) at fps."""
    duration = max(0.0, t_end - t_start)
    if duration <= 1e-9 or fps <= 0:
        return []
    n_frames = max(1, int(round(duration * fps)))
    return _sample_times(t_start, t_end, n_frames)


def _write_frames(
    cap: cv2.VideoCapture,
    out_dir: Path,
    timestamps: Sequence[float],
    label: str | None = None,
    image_ext: str = "jpg",
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in out_dir.glob(f"frame_*.{image_ext}"):
        old_path.unlink()
    frame_paths = []
    total = len(timestamps)
    for frame_i, timestamp in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        if label:
            text = label.format(i=frame_i + 1, n=total)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            margin = 12
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(
                frame,
                (0, 0),
                (text_w + 2 * margin, text_h + baseline + 2 * margin),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                frame,
                text,
                (margin, margin + text_h),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        path = out_dir / f"frame_{frame_i:02d}.{image_ext}"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write frame: {path}")
        frame_paths.append(str(path))
    return frame_paths


def extract_frames(
    video_path: str | Path,
    window: Window,
    frames_root: str | Path,
    n_frames_per_window: int = 8,
    image_ext: str = "jpg",
) -> List[str]:
    """Save uniformly sampled frames for one W window and return paths."""
    window_idx, t_start, t_end = window
    out_dir = Path(frames_root) / f"win_{window_idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_paths = []
    for frame_i, timestamp in enumerate(_sample_times(t_start, t_end, n_frames_per_window)):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{frame_i:02d}.{image_ext}"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write frame: {path}")
        frame_paths.append(str(path))
    cap.release()
    return frame_paths


def extract_context_frame_groups(
    video_path: str | Path,
    window_idx: int,
    t_start: float,
    t_end: float,
    output_window_dir: str | Path,
    target_fps: float,
    pre_context_sec: float,
    post_context_sec: float,
    context_fps: float,
    context_mode: str,
    overlay_frame_labels: bool = True,
) -> dict:
    """Save before/target/after W frame groups and return grouped paths/counts."""
    if context_mode not in {"both", "past", "none"}:
        raise ValueError(f"Unsupported context_mode: {context_mode}")

    video_duration = video_duration_sec(video_path)
    before_start = max(0.0, float(t_start) - float(pre_context_sec))
    before_end = float(t_start)
    target_start = float(t_start)
    target_end = float(t_end)
    after_start = float(t_end)
    after_end = min(video_duration, float(t_end) + float(post_context_sec))

    if context_mode == "none":
        before_times: Sequence[float] = []
        after_times: Sequence[float] = []
    elif context_mode == "past":
        before_times = _sample_times_at_fps(before_start, before_end, context_fps)
        after_times = []
    else:
        before_times = _sample_times_at_fps(before_start, before_end, context_fps)
        after_times = _sample_times_at_fps(after_start, after_end, context_fps)
    target_times = _sample_times_at_fps(target_start, target_end, target_fps)

    out_dir = Path(output_window_dir)
    before_dir = out_dir / "before"
    target_dir = out_dir / "target"
    after_dir = out_dir / "after"
    before_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    before_label = "BEFORE CONTEXT - DO NOT LABEL" if overlay_frame_labels else None
    target_label = "TARGET FRAME {i}/{n} - LABEL THIS INTERVAL" if overlay_frame_labels else None
    after_label = "AFTER CONTEXT - DO NOT LABEL" if overlay_frame_labels else None
    before_paths = _write_frames(cap, before_dir, before_times, before_label)
    target_paths = _write_frames(cap, target_dir, target_times, target_label)
    after_paths = _write_frames(cap, after_dir, after_times, after_label)
    cap.release()

    return {
        "before_frame_paths": before_paths,
        "target_frame_paths": target_paths,
        "after_frame_paths": after_paths,
        "before_frames_path": str(before_dir),
        "target_frames_path": str(target_dir),
        "after_frames_path": str(after_dir),
        "before_context_frame_count": len(before_paths),
        "target_frame_count": len(target_paths),
        "after_context_frame_count": len(after_paths),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--win-sec", type=float, default=3)
    parser.add_argument("--stride-sec", type=float, default=3)
    parser.add_argument("--n-frames-per-window", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    windows = make_windows(args.video_path, args.win_sec, args.stride_sec)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    for window in windows:
        paths = extract_frames(
            args.video_path,
            window,
            args.frames_root,
            args.n_frames_per_window,
        )
        print(f"W window {window[0]} frames={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
