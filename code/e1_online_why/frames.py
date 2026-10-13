"""E1 frame extraction using only query time and model-selected intervals."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from code.e0.dataset import atomic_write_json, sha256_file, stable_hash
from code.e0.frames import extract_or_reuse_frames


def validate_interval(start_sec: float, end_sec: float, query_time_sec: float) -> None:
    values = (start_sec, end_sec, query_time_sec)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError("video interval values must be finite numbers")
    if not (start_sec >= 0.0 and start_sec < end_sec and end_sec <= query_time_sec + 1e-6):
        raise ValueError(
            f"invalid requested interval [{start_sec}, {end_sec}] for query time {query_time_sec}"
        )


def interval_record(
    query: Mapping[str, Any], start_sec: float, end_sec: float, view_index: int
) -> dict[str, Any]:
    validate_interval(start_sec, end_sec, float(query["query_time_sec"]))
    return {
        "question_id": f"{query['question_id']}:view{view_index}",
        "source_video_id": query["source_video_id"],
        "source_video_path": query["source_video_path"],
        "requested_start": f"{start_sec:.6f}",
        "requested_end": f"{end_sec:.6f}",
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
    }


def extract_view(
    query: Mapping[str, Any],
    start_sec: float,
    end_sec: float,
    view_index: int,
    cache_root: Path,
    frame_settings: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    record = interval_record(query, start_sec, end_sec, view_index)
    frame_record, reused = extract_or_reuse_frames(record, cache_root, frame_settings)
    query_time = float(query["query_time_sec"])
    for frame in frame_record["frames"]:
        timestamp = float(frame["timestamp_sec"])
        if not start_sec - 1e-6 <= timestamp <= end_sec + 1e-6:
            raise ValueError(f"decoded frame {timestamp} lies outside requested interval")
        if timestamp > query_time + 1e-6:
            raise ValueError(f"decoded frame {timestamp} lies after query time {query_time}")
    return frame_record, reused


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_or_reuse_contact_sheets(
    frame_record: Mapping[str, Any], cache_root: Path
) -> tuple[dict[str, Any], bool]:
    """Represent one 16-frame view as at most four timestamp-labeled images."""

    identity = {
        "ordered_frame_payload_hash": frame_record["ordered_frame_payload_hash"],
        "layout": "2x2_chronological_v1",
        "canvas": [1280, 1280],
        "tiles_per_sheet": 4,
        "label": "frame index and source timestamp seconds",
        "jpeg_quality": 92,
    }
    cache_id = stable_hash(identity)
    destination = cache_root / cache_id
    record_path = destination / "record.json"
    if record_path.is_file():
        cached = json.loads(record_path.read_text(encoding="utf-8"))
        if cached.get("cache_id") != cache_id:
            raise ValueError(f"incompatible contact-sheet cache at {record_path}")
        for sheet in cached.get("contact_sheets", []):
            path = Path(sheet["path"])
            if not path.is_file() or sha256_file(path) != sheet["sha256"]:
                raise ValueError(f"corrupt contact sheet: {path}")
        return cached, True

    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="e1-contact-", dir="/tmp"))
    try:
        frames = list(frame_record["frames"])
        sheets: list[dict[str, Any]] = []
        font = _font(22)
        for sheet_index, offset in enumerate(range(0, len(frames), 4)):
            chunk = frames[offset : offset + 4]
            canvas = Image.new("RGB", (1280, 1280), "black")
            draw = ImageDraw.Draw(canvas)
            tiles: list[dict[str, Any]] = []
            for local_index, frame in enumerate(chunk):
                row, column = divmod(local_index, 2)
                left, top = column * 640, row * 640
                label = (
                    f"Frame {offset + local_index + 1:02d}  |  "
                    f"t={float(frame['timestamp_sec']):.3f} s"
                )
                draw.rectangle((left, top, left + 639, top + 39), fill="black")
                draw.text((left + 12, top + 8), label, fill="white", font=font)
                with Image.open(frame["path"]) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((640, 600), Image.Resampling.LANCZOS)
                    x = left + (640 - image.width) // 2
                    y = top + 40 + (600 - image.height) // 2
                    canvas.paste(image, (x, y))
                tiles.append(
                    {
                        "tile": local_index + 1,
                        "frame_index": offset + local_index,
                        "timestamp_sec": float(frame["timestamp_sec"]),
                        "source_frame_sha256": frame["sha256"],
                    }
                )
            temp_path = temporary / f"contact_{sheet_index:02d}.jpg"
            canvas.save(temp_path, format="JPEG", quality=92, subsampling=0, optimize=True)
            output = destination / temp_path.name
            shutil.move(str(temp_path), output)
            sheets.append(
                {
                    "index": sheet_index,
                    "path": str(output.resolve()),
                    "sha256": sha256_file(output),
                    "tiles": tiles,
                }
            )
        payload = {
            "cache_id": cache_id,
            **identity,
            "contact_sheets": sheets,
        }
        atomic_write_json(record_path, payload)
        return payload, False
    except Exception:
        if not record_path.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
