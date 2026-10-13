"""Timestamp-bounded source-frame extraction for E0 clips only."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .dataset import atomic_write_json, sha256_file, stable_hash


def video_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return {
        "method": "sha256(size,mtime_ns,first_1MiB,last_1MiB)",
        "value": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _frame_pts(path: Path, start: float, end: float) -> list[float]:
    probe_start = max(0.0, start - 5.0)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-read_intervals",
        f"{probe_start:.6f}%{end + 0.2:.6f}",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    points = sorted(
        {
            float(frame["best_effort_timestamp_time"])
            for frame in payload.get("frames", [])
            if "best_effort_timestamp_time" in frame
            and start - 1e-6 <= float(frame["best_effort_timestamp_time"]) <= end + 1e-6
        }
    )
    return points


def uniform_indices(population: int, target: int) -> list[int]:
    if population <= 0 or target <= 0:
        return []
    if population <= target:
        return list(range(population))
    # Endpoint-inclusive integer selection without floating duplicate indices.
    return [(index * (population - 1)) // (target - 1) for index in range(target)]


def _extract_selected_images(
    path: Path, start: float, selected_indices: Sequence[int], temporary: Path
) -> list[Path]:
    if not selected_indices:
        return []
    # Input seeking is accurate for transcoding: ffmpeg seeks to the preceding
    # keyframe and discards decoded frames before start. Selecting by relative
    # decoded index then emits only the PTS positions chosen from ffprobe, avoiding
    # expensive PNG encoding of every frame in a long action interval.
    expression = "+".join(f"eq(n\\,{index})" for index in selected_indices)
    select = f"select='{expression}'"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(path),
        "-vf",
        select,
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(len(selected_indices)),
        str(temporary / "source_%06d.png"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return sorted(temporary.glob("source_*.png"))


def _preprocess(source: Path, destination: Path, settings: Mapping[str, Any]) -> None:
    max_edge = int(settings["max_edge"])
    quality = int(settings["jpeg_quality"])
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=quality, subsampling=0, optimize=True)


def frame_identity(record: Mapping[str, Any], fingerprint: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "video": fingerprint,
            "start_sec": record["start_sec"],
            "end_sec": record["end_sec"],
            "sampling": {"method": "uniform_distinct_pts_v1", "target_count": settings["count"]},
            "preprocessing": settings,
        }
    )


def extract_or_reuse_frames(
    record: Mapping[str, Any], cache_root: Path, settings: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    video_path = Path(record["source_video_path"])
    fingerprint = video_fingerprint(video_path)
    cache_id = frame_identity(record, fingerprint, settings)
    destination = cache_root / cache_id
    record_path = destination / "record.json"
    if record_path.is_file():
        cached = json.loads(record_path.read_text(encoding="utf-8"))
        if cached.get("cache_id") != cache_id:
            raise ValueError(f"incompatible frame cache record at {record_path}")
        for frame in cached.get("frames", []):
            frame_path = Path(frame["path"])
            if not frame_path.is_file() or sha256_file(frame_path) != frame["sha256"]:
                raise ValueError(f"corrupt cached frame: {frame_path}")
            timestamp = float(frame["timestamp_sec"])
            if not float(record["start_sec"]) - 1e-6 <= timestamp <= float(record["end_sec"]) + 1e-6:
                raise ValueError(f"cached frame outside requested interval: {timestamp}")
        return cached, True

    points = _frame_pts(video_path, float(record["start_sec"]), float(record["end_sec"]))
    if not points:
        raise RuntimeError(f"no source frame PTS inside interval for {record['question_id']}")
    selected_indices = uniform_indices(len(points), int(settings["count"]))
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="e0-extract-", dir="/tmp"))
    try:
        extracted = _extract_selected_images(
            video_path, float(record["start_sec"]), selected_indices, temporary
        )
        if len(extracted) != len(selected_indices):
            raise RuntimeError(
                f"ffmpeg selected-frame mismatch for {record['question_id']}: "
                f"{len(selected_indices)} requested versus {len(extracted)} decoded frames"
            )
        frame_rows: list[dict[str, Any]] = []
        for output_index, (source_index, extracted_path) in enumerate(zip(selected_indices, extracted)):
            destination_path = destination / f"frame_{output_index:02d}.jpg"
            _preprocess(extracted_path, destination_path, settings)
            timestamp = points[source_index]
            if not float(record["start_sec"]) - 1e-6 <= timestamp <= float(record["end_sec"]) + 1e-6:
                raise AssertionError("selected an out-of-interval frame")
            frame_rows.append(
                {
                    "index": output_index,
                    "source_frame_index_within_interval": source_index,
                    "timestamp_sec": timestamp,
                    "path": str(destination_path.resolve()),
                    "sha256": sha256_file(destination_path),
                }
            )
        ordered_payload_hash = stable_hash(
            [{"timestamp_sec": row["timestamp_sec"], "sha256": row["sha256"]} for row in frame_rows]
        )
        payload = {
            "cache_id": cache_id,
            "question_id_first_requester": record["question_id"],
            "source_video_path": str(video_path.resolve()),
            "source_video_id": record["source_video_id"],
            "source_video_fingerprint": fingerprint,
            "requested_start": record["requested_start"],
            "requested_end": record["requested_end"],
            "start_sec": record["start_sec"],
            "end_sec": record["end_sec"],
            "timestamp_source": "ffprobe best_effort_timestamp_time",
            "sampling_method": "uniform_distinct_pts_v1",
            "preprocessing": dict(settings),
            "available_source_frame_count": len(points),
            "requested_frame_count": int(settings["count"]),
            "actual_unique_frame_count": len(frame_rows),
            "used_all_available_frames": len(points) < int(settings["count"]),
            "ordered_frame_payload_hash": ordered_payload_hash,
            "frames": frame_rows,
        }
        atomic_write_json(record_path, payload)
        return payload, False
    except Exception:
        # Only remove this not-yet-valid cache identity; unrelated caches are untouched.
        if not record_path.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def extract_many(
    records: Sequence[Mapping[str, Any]], cache_root: Path, settings: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    result: dict[str, dict[str, Any]] = {}
    reused = 0
    created = 0
    by_clip: dict[str, dict[str, Any]] = {}
    for record in records:
        clip_key = str(record["source_clip_key"])
        if clip_key in by_clip:
            frame_record = by_clip[clip_key]
            was_reused = True
        else:
            frame_record, was_reused = extract_or_reuse_frames(record, cache_root, settings)
            by_clip[clip_key] = frame_record
        result[str(record["question_id"])] = frame_record
        reused += int(was_reused)
        created += int(not was_reused)
    return result, {"reused": reused, "created": created}
