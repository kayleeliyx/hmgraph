"""Full-frame selection for L1 multimodal verification."""



import os
from pathlib import Path


def _select_l1_reasoning_frames_base(video_path: str, segment: dict, max_frames: int = 24, output_dir: str | Path | None = None) -> list[dict]:
    import cv2

    configured_root = output_dir or os.environ.get("L1_REASONING_FRAME_OUTPUT_DIR")
    if configured_root is None:
        configured_root = "l1_reasoning_frames"
    out_root = Path(configured_root)
    out_dir = out_root / segment["segment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    frame_budget = _frame_budget(segment, max_frames)
    times = _limit_times(_sample_times(segment), frame_budget)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    records = []
    for idx, (timestamp, role) in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{idx:02d}_{role}_{timestamp:.2f}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write frame: {path}")
        records.append({"frame_id": f"{segment['segment_id']}_frame_{idx:02d}", "timestamp_sec": round(float(timestamp), 3), "role": role, "image_path": str(path), "frame_budget": frame_budget})
    cap.release()
    return records


def _sample_times(segment: dict) -> list[tuple[float, str]]:
    ct0, t0, t1, ct1 = map(float, (segment["context_t_start"], segment["t_start"], segment["t_end"], segment["context_t_end"]))
    times: list[tuple[float, str]] = []
    before = _fps_times(ct0, t0, 1.5)
    after = _fps_times(t1, ct1, 1.5)
    if before:
        times.append((before[-1], "before_context"))
    if after:
        times.append((after[0], "after_context"))
    core = _fps_times(t0, t1, _core_sampling_fps(segment))
    boundaries = [t0, (t0 + t1) / 2.0, max(t0, t1 - 0.05)]
    for window_idx in segment.get("source_window_indices", []):
        boundaries.extend([float(window_idx), float(window_idx) + 1.0])
    for t in boundaries:
        if ct0 <= t <= ct1 and not any(abs(t - existing) < 0.03 for existing in core):
            core.append(t)
    times.extend((t, "core_segment") for t in sorted(core) if t0 <= t <= t1)
    times.extend((t, "before_context") for t in before[:-1])
    times.extend((t, "after_context") for t in after[1:])
    return _dedup(times)


def _frame_budget(segment: dict, max_frames: int) -> int:
    requested = segment.get("recommended_frame_budget")
    try:
        requested_i = int(requested)
    except Exception:
        requested_i = max_frames
    return max(4, min(int(max_frames), requested_i))


def _core_sampling_fps(segment: dict) -> float:
    diagnostics = segment.get("w_grouping_diagnostics") or {}
    if diagnostics.get("needs_dense_visual_verification"):
        return 6.0
    duration = float(segment["t_end"]) - float(segment["t_start"])
    return 5.0 if duration > 4.0 else 4.0


def _limit_times(times: list[tuple[float, str]], max_frames: int) -> list[tuple[float, str]]:
    if len(times) <= max_frames:
        return sorted(times)
    core = [x for x in times if x[1] == "core_segment"]
    context = [x for x in times if x[1] != "core_segment"]
    keep_context = min(len(context), max(4, max_frames // 4))
    keep_core = max_frames - keep_context
    return sorted(_even_pick(core, keep_core) + _even_pick(context, keep_context))


def _fps_times(start: float, end: float, fps: float) -> list[float]:
    if end <= start:
        return []
    n = max(1, int(round((end - start) * fps)))
    if n == 1:
        return [(start + end) / 2.0]
    eps = min(0.04, (end - start) / 100.0)
    lo, hi = start + eps, end - eps
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def _dedup(times: list[tuple[float, str]]) -> list[tuple[float, str]]:
    out, seen = [], set()
    for t, role in sorted(times):
        key = round(max(0.0, t), 2)
        if key not in seen:
            seen.add(key)
            out.append((key, role))
    return out


def _even_pick(items: list[tuple[float, str]], n: int) -> list[tuple[float, str]]:
    if n <= 0:
        return []
    if len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]
    idxs = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[i] for i in sorted(set(idxs))]

def select_l1_reasoning_frames(video_path: str, segment: dict, max_frames: int = 24, output_dir=None) -> list[dict]:
    return _select_l1_reasoning_frames_base(video_path, segment, max_frames, output_dir=output_dir)
