"""Resumable frame cache, with actual decoder PTS provenance for every JPEG."""
from __future__ import annotations

import argparse
import bisect
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
from PIL import Image

from code.e0.dataset import atomic_write_json, sha256_file, stable_hash
from code.e0.frames import video_fingerprint
from .build import DEFAULT_OUT, FRAME_SETTINGS, PARTITIONS, read_jsonl, text_fingerprint


def sample_records(out):
    result = []
    for p in PARTITIONS:
        rows = list(read_jsonl(out / "text" / f"{p}.jsonl"))
        for task in ("explain", "track"):
            candidates = [r for r in rows if r["task"] == task and r["query_time_sec"] >= 5]
            candidates.sort(key=lambda r: stable_hash(["preview_v1", r["id"]]))
            result.append(candidates[0])
        if p in ("initial_sft", "test"):
            candidates = [r for r in rows if r["task"] == "explain" and r["id"] not in {a["id"] for a in result}]
            candidates.sort(key=lambda r: stable_hash(["preview_v1", r["id"]]))
            result.append(candidates[0])
    return result


def extract_video(job):
    video, indices, out, threads = job
    started = time.monotonic()
    root = Path(out) / "frames" / video["video_id"] / video["cache_id"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "record.json"
    if video_fingerprint(Path(video["path"])) != video["fingerprint"]:
        raise ValueError(f"Source video changed: {video['path']}")
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"video_id": video["video_id"], "cache_id": video["cache_id"], "source_video_path": video["path"], "source_video_fingerprint": video["fingerprint"], "preprocessing": FRAME_SETTINGS, "timestamp_source": "PyAV decoded frame.pts * frame.time_base (original MP4 PTS)", "frames": {}}
    if existing["cache_id"] != video["cache_id"] or existing["preprocessing"] != FRAME_SETTINGS:
        raise ValueError("Cache identity or preprocessing mismatch")
    missing, reused = [], 0
    for index in indices:
        saved = existing["frames"].get(str(index))
        path = root / f"{index:08d}.jpg"
        if saved and path.is_file() and sha256_file(path) == saved["sha256"]:
            if saved["timestamp_sec"] != index / 30 or saved["source_pts"] != index * 512:
                raise ValueError("Cached source timestamp mismatch")
            reused += 1
        else:
            missing.append(index)
    created, decoded = 0, 0
    if missing:
        with av.open(video["path"]) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.thread_count = threads
            pos = 0
            while pos < len(missing):
                # Seek to a preceding keyframe. Restart only across large gaps,
                # so bulk extraction is mostly one decode pass per source video.
                container.seek(missing[pos] * 512, stream=stream, backward=True)
                jumped = False
                for frame in container.decode(stream):
                    decoded += 1
                    if frame.pts is None:
                        raise ValueError("Decoder emitted a frame without PTS")
                    timestamp = float(frame.pts * frame.time_base)
                    index = round(timestamp * 30)
                    if timestamp != index / 30 or frame.pts != index * 512:
                        raise ValueError(f"Inspected CFR assumption failed in actual decode: {video['video_id']} PTS={frame.pts}")
                    if index < missing[pos]:
                        continue
                    if index > missing[pos]:
                        raise ValueError(f"Decoder skipped requested source frame {missing[pos]} in {video['video_id']}")
                    scale = FRAME_SETTINGS["max_edge"] / max(frame.width, frame.height)
                    width, height = round(frame.width * min(1., scale)), round(frame.height * min(1., scale))
                    image = frame.reformat(width=width, height=height, format="rgb24", interpolation=av.video.reformatter.Interpolation.LANCZOS).to_image()
                    path = root / f"{index:08d}.jpg"
                    temporary = root / f".{index:08d}.{os.getpid()}.tmp"
                    image.save(temporary, format="JPEG", quality=FRAME_SETTINGS["jpeg_quality"], subsampling=0, optimize=True)
                    os.replace(temporary, path)
                    existing["frames"][str(index)] = {"source_frame_index": index, "source_pts": frame.pts, "source_time_base": str(frame.time_base), "timestamp_sec": timestamp, "path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size, "width": width, "height": height}
                    created += 1
                    pos += 1
                    if created % 256 == 0:
                        atomic_write_json(manifest_path, existing)
                    if pos == len(missing):
                        break
                    if missing[pos] - index > 300:
                        jumped = True
                        break
                if pos < len(missing) and not jumped:
                    raise ValueError(f"End of video before all requested frames: {video['video_id']}")
        atomic_write_json(manifest_path, existing)
    return {"video_id": video["video_id"], "created": created, "reused": reused, "requested": len(indices), "decoded": decoded, "seconds": round(time.monotonic() - started, 3)}


def extract(out, sample=False, workers=16, threads=2):
    inventory = json.loads((out / "video_inventory.json").read_text())
    requests = json.loads((out / "frame_requests.json").read_text())
    if sample:
        requests = {}
        for r in sample_records(out):
            requests.setdefault(r["video_id"], set()).update(f["source_frame_index"] for f in r["frames"])
        requests = {v: sorted(indices) for v, indices in requests.items()}
    else:
        sample_audit = out / "sample_audit.json"
        if not sample_audit.exists() or not json.loads(sample_audit.read_text()).get("passed"):
            raise ValueError("Bulk extraction requires a passed small-sample record audit first")
        fingerprint = text_fingerprint(out)
        for name in ("text_audit.json", "sample_audit.json"):
            if json.loads((out / name).read_text()).get("dataset_fingerprint") != fingerprint:
                raise ValueError(f"{name} does not validate the current text records; rerun the text/sample audits")
    jobs = [(inventory[v], indices, str(out), threads) for v, indices in sorted(requests.items())]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_video, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed_videos": len(results), "total_videos": len(jobs), **result}), flush=True)
            atomic_write_json(out / ("sample_extraction_progress.json" if sample else "extraction_progress.json"), results)
    atomic_write_json(out / ("sample_extraction_summary.json" if sample else "extraction_summary.json"), {"videos": results, "created": sum(r["created"] for r in results), "reused": sum(r["reused"] for r in results), "requested": sum(r["requested"] for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    extract(args.output.resolve(), args.sample, args.workers, args.threads)


if __name__ == "__main__":
    main()
