#!/usr/bin/env python3
"""Freeze the released action, activity and purpose annotations for inspection."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

VERSION = "hdepic_source_timeline_v1"
ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / "datasets" / "hdepic_sft_v2_repaired"
DEFAULT_ANNOTATIONS = Path("/data/kailaic/HDEPIC/data/hd-epic-annotations")
VIDEO_ID_REPAIRS = {"P09-2024623-154547": "P09-20240623-154547"}


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def valid_interval(start, end, duration):
    return (start is not None and end is not None and duration is not None
            and 0 <= start <= end <= duration)


def union_duration(intervals):
    intervals = sorted((start, end) for start, end in intervals
                       if start is not None and end is not None and 0 <= start <= end)
    total, left, right = 0.0, None, None
    for start, end in intervals:
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


def _classes(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {int(row["id"]): row["key"] for row in csv.DictReader(handle)}


def build(reason_labels, inventory_path, activities_dir, class_dir, output):
    """Preserve all source intervals; flag invalid bounds without truncating them."""
    reason_labels, inventory_path = Path(reason_labels), Path(inventory_path)
    activities_dir, class_dir, output = Path(activities_dir), Path(class_dir), Path(output)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    verb_path, noun_path = class_dir / "HD_EPIC_verb_classes.csv", class_dir / "HD_EPIC_noun_classes.csv"
    classes = {"verb": _classes(verb_path), "noun": _classes(noun_path)}
    hashes = {str(path.resolve()): sha256(path) for path in (reason_labels, inventory_path, verb_path, noun_path)}
    original_narrations = class_dir / "HD_EPIC_Narrations.pkl"
    if original_narrations.is_file():
        hashes[str(original_narrations.resolve())] = sha256(original_narrations)
    split_manifest = reason_labels.parent / "split_manifest.json"
    annotation_lineage = {}
    if split_manifest.is_file():
        hashes[str(split_manifest.resolve())] = sha256(split_manifest)
        annotation_lineage["original_narrations_sha256"] = json.loads(split_manifest.read_text()).get("narrations_sha256")
    videos = {identity: {"id": identity, "participant": entry.get("participant", identity.split("-")[0]),
                         "partition": None, "duration_sec": number(entry.get("duration_sec")),
                         "actions": [], "goals": [], "why": [], "source_complete": True,
                         "source_note": "Complete released annotations, including overlaps and gaps. WHY intervals identify the narrated action, not how long a purpose lasts.",
                         "issues": []}
              for identity, entry in inventory.items()}
    issues, action_ids, source_action_count, status_counts = [], set(), 0, Counter()

    def issue(category, video_id=None, **details):
        value = {"category": category, "video_id": video_id, **details}
        issues.append(value)
        if video_id in videos:
            videos[video_id]["issues"].append(value)

    with reason_labels.open(encoding="utf-8") as handle:
        for line, encoded in enumerate(handle, 1):
            if not encoded.strip():
                continue
            row = json.loads(encoded)
            source_action_count += 1
            identity, video_id = row["narration_id"], row["video_id"]
            if identity in action_ids:
                raise ValueError(f"Duplicate narration ID: {identity}")
            action_ids.add(identity)
            if video_id not in videos:
                issue("action_video_not_in_inventory", video_id, action_id=identity, source_line=line, source_record=row)
                continue
            video = videos[video_id]
            partition = row.get("partition")
            if video["partition"] is not None and video["partition"] != partition:
                issue("conflicting_video_partition", video_id, action_id=identity, partition=partition)
            elif video["partition"] is None:
                video["partition"] = partition
            target = row.get("target") or {}
            start, end = number(target.get("start_sec")), number(target.get("end_sec"))
            pairs = target.get("main_action_class_ids") or []
            pair = pairs[0] if len(pairs) == 1 else []
            verb_id = pair[0] if pair else None
            noun_id = pair[1] if len(pair) > 1 else None
            valid = valid_interval(start, end, video["duration_sec"])
            action = {"id": identity, "start_sec": start, "end_sec": end,
                      "text": target.get("action") or None, "verb_id": verb_id, "noun_id": noun_id,
                      "verb": classes["verb"].get(verb_id), "noun": classes["noun"].get(noun_id),
                      "class_pairs": pairs, "main_actions": target.get("main_actions") or [],
                      "class_label_status": target.get("class_label_status", "unavailable"),
                      "narration": row.get("original_narration"), "valid_interval": valid}
            video["actions"].append(action)
            if not valid:
                issue("action_interval_out_of_bounds", video_id, action_id=identity,
                      original_start_sec=target.get("start_sec"), original_end_sec=target.get("end_sec"), duration_sec=video["duration_sec"])
            if not target.get("action") or not pairs:
                issue("missing_canonical_main_action", video_id, action_id=identity)
            elif verb_id is None or noun_id is None:
                issue("partial_or_multiple_main_action_classes", video_id, action_id=identity, class_pairs=pairs)
            status = row.get("status", "unknown")
            status_counts[status] += 1
            if row.get("reason_text"):
                why = {"id": identity + ":why", "action_id": identity, "start_sec": start, "end_sec": end,
                       "text": row["reason_text"], "status": status, "accepted": status.startswith("accepted"),
                       "narration": row.get("original_narration"), "char_start": row.get("char_start"),
                       "char_end": row.get("char_end"), "extraction_rule": row.get("extraction_rule"),
                       "rejection_category": row.get("rejection_category"), "valid_interval": valid,
                       "interval_meaning": "target_action_interval; not purpose duration",
                       "human_validated": False}
                video["why"].append(why)
    activity_paths = sorted(activities_dir.glob("*.csv"))
    if not activity_paths:
        raise ValueError(f"No activity CSVs found: {activities_dir}")
    goal_ids, source_goal_count = set(), 0
    for path in activity_paths:
        hashes[str(path.resolve())] = sha256(path)
        with path.open(encoding="utf-8", newline="") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                source_goal_count += 1
                identity = f"{path.name}:{line}"
                if identity in goal_ids:
                    raise ValueError(f"Duplicate activity ID: {identity}")
                goal_ids.add(identity)
                original_video_id = row["video_id"]
                video_id = VIDEO_ID_REPAIRS.get(original_video_id, original_video_id)
                if video_id != original_video_id:
                    issue("activity_video_id_typo_repaired", video_id, id=identity, original_video_id=original_video_id)
                if video_id not in videos:
                    issue("unmatched_activity_video_id", original_video_id, id=identity, source_record=row)
                    continue
                video = videos[video_id]
                start = number(row.get("start_time"))
                end = video["duration_sec"] if row.get("end_time", "").strip().lower() == "end" else number(row.get("end_time"))
                valid = valid_interval(start, end, video["duration_sec"])
                goal = {"id": identity, "start_sec": start, "end_sec": end,
                        "text": row.get("high_level_activity_label", "").strip(), "recipe_id": row.get("recipe_id") or None,
                        "original_video_id": original_video_id, "resolved_video_id": video_id,
                        "original_start_time": row.get("start_time"), "original_end_time": row.get("end_time"),
                        "scope": "as_annotated", "valid_interval": valid}
                video["goals"].append(goal)
                if not valid:
                    issue("reversed_activity_interval" if start is not None and end is not None and end < start else "activity_interval_out_of_bounds", video_id, id=identity, original_start_time=row.get("start_time"),
                          original_end_time=row.get("end_time"), duration_sec=video["duration_sec"])
                if not goal["text"]:
                    issue("empty_activity_label", video_id, id=identity)
    for video in videos.values():
        for kind in ("actions", "goals", "why"):
            video[kind].sort(key=lambda row: (row["start_sec"] is None, row["start_sec"] or 0,
                                               row["end_sec"] or 0, row["id"]))
        if not video["actions"]:
            video["source_complete"] = False
            issue("inventory_video_has_no_action_annotations", video["id"])
    result = {"version": VERSION, "provenance": {
        "source_sha256": hashes, "class_names": "released official verb/noun class keys",
        "annotation_lineage": annotation_lineage,
        "why_source": "Frozen annotation extraction; accepted candidates remain pending human review.",
        "goal_source": "Released activity CSVs, unmodified original scope and intervals.",
        "query_scope": "Human review points and model forecasts are supplied at runtime; source asset includes no model predictions or judgments.",
        "video_id_repairs": VIDEO_ID_REPAIRS},
        "videos": videos, "audit": {"source_action_rows": source_action_count, "retained_actions": sum(len(v["actions"]) for v in videos.values()),
        "source_goal_rows": source_goal_count, "retained_goals": sum(len(v["goals"]) for v in videos.values()),
        "why_candidates": sum(len(v["why"]) for v in videos.values()),
        "accepted_why": sum(sum(w["accepted"] for w in v["why"]) for v in videos.values()),
        "reason_extraction_status_counts": dict(status_counts), "video_count": len(videos),
        "issue_counts": dict(Counter(x["category"] for x in issues)), "issues": issues}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason-labels", type=Path, default=DEFAULT_DATASET / "reason_labels.jsonl")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_DATASET / "video_inventory.json")
    parser.add_argument("--activities", type=Path, default=DEFAULT_ANNOTATIONS / "high-level" / "activities")
    parser.add_argument("--classes", type=Path, default=DEFAULT_ANNOTATIONS / "narrations-and-action-segments")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "timeline_annotations.json.gz")
    args = parser.parse_args()
    result = build(args.reason_labels, args.inventory, args.activities, args.classes, args.output)
    print(json.dumps({"asset": str(args.output.resolve()), "sha256": sha256(args.output),
                      **{key: value for key, value in result["audit"].items() if key != "issues"}}))


if __name__ == "__main__":
    main()
