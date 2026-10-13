"""Frame provenance helpers for the two local conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from code.e0.dataset import sha256_file
from code.e1_online_why.frames import extract_view, validate_interval


def load_e0_action_index(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict) or len(items) != 100:
        raise ValueError("E0 action-frame index must contain the exact 100 dev items")
    return items


def action_clip_batch(question_id: str, index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    item = index.get(question_id)
    if item is None:
        raise KeyError(f"missing E0 C1 frame record for {question_id}")
    path = Path(str(item["record_path"]))
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("cache_id") != item.get("cache_id"):
        raise ValueError(f"E0 frame-index mismatch for {question_id}")
    frames = []
    for frame in record.get("frames", []):
        frame_path = Path(str(frame["path"]))
        if not frame_path.is_file() or sha256_file(frame_path) != frame["sha256"]:
            raise ValueError(f"missing or corrupt saved E0 frame: {frame_path}")
        frames.append(dict(frame))
    if not frames or len(frames) > 16:
        raise ValueError(f"unexpected E0 action frame count for {question_id}: {len(frames)}")
    return {
        "cache_id": record["cache_id"],
        "cache_hit": True,
        "source": "exact_saved_e0_c1_action_clip_frames",
        "source_record_path": str(path.resolve()),
        "source_video_id": record["source_video_id"],
        "start_sec": float(record["start_sec"]),
        "end_sec": float(record["end_sec"]),
        "requested_frame_count": int(record["requested_frame_count"]),
        "actual_unique_frame_count": int(record["actual_unique_frame_count"]),
        "used_all_available_frames": bool(record["used_all_available_frames"]),
        "ordered_frame_payload_hash": record["ordered_frame_payload_hash"],
        "frames": frames,
    }


def local_view(
    query: Mapping[str, Any],
    start_sec: float,
    end_sec: float,
    view_index: int,
    cache_root: Path,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    record, reused = extract_view(
        query, start_sec, end_sec, view_index, cache_root, settings
    )
    return {
        "cache_id": record["cache_id"],
        "cache_hit": reused,
        "source": "source_video_model_selected_or_fixed_interval",
        "source_video_id": record["source_video_id"],
        "start_sec": float(record["start_sec"]),
        "end_sec": float(record["end_sec"]),
        "requested_frame_count": int(record["requested_frame_count"]),
        "actual_unique_frame_count": int(record["actual_unique_frame_count"]),
        "used_all_available_frames": bool(record["used_all_available_frames"]),
        "ordered_frame_payload_hash": record["ordered_frame_payload_hash"],
        "frames": [dict(frame) for frame in record["frames"]],
    }, reused


__all__ = ["action_clip_batch", "load_e0_action_index", "local_view", "validate_interval"]

