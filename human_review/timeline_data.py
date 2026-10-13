"""Full source timelines plus sparse evaluation queries and saved human labels.

The runtime uses only the standard library and a frozen gzip asset. Source
annotation intervals, generated forecasts and human review points stay distinct.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "hdepic_source_timeline_v1"


def union_duration(intervals):
    valid = sorted((start, end) for start, end in intervals
                   if isinstance(start, (int, float)) and isinstance(end, (int, float))
                   and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end)
    total, left, right = 0.0, None, None
    for start, end in valid:
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


def _coverage(rows, duration):
    valid = [row for row in rows if row.get("valid_interval", True)]
    covered = union_duration([(row.get("start_sec"), row.get("end_sec")) for row in valid])
    return {"union_duration_sec": covered,
            "coverage_fraction": covered / duration if duration and duration > 0 else None,
            "uncovered_duration_sec": max(0, duration - covered) if duration is not None else None,
            "excluded_invalid_intervals": len(rows) - len(valid)}


class TimelineCatalog:
    def __init__(self, review_catalog):
        self.catalog = review_catalog
        self.warnings = []
        self.videos = {}
        self.provenance = {}
        self.audit = {}
        self._inventory = {}
        self._video_cache = {}
        inventory_value = review_catalog.config.get("timeline_video_inventory", review_catalog.config.get("video_inventory"))
        self._inventory_path = review_catalog._path(inventory_value) if inventory_value else None
        if self._inventory_path and self._inventory_path.is_file():
            self._inventory = json.loads(self._inventory_path.read_text(encoding="utf-8"))
        value = review_catalog.config.get("timeline_annotations")
        asset_path = review_catalog._path(value) if value else None
        if asset_path and asset_path.is_file():
            with gzip.open(asset_path, "rt", encoding="utf-8") as handle:
                asset = json.load(handle)
            if asset.get("version") != VERSION:
                raise ValueError("Unsupported timeline source asset version")
            self.videos = asset["videos"]
            self.provenance = asset.get("provenance", {})
            self.audit = asset.get("audit", {})
            for identity, video in self.videos.items():
                if video.get("id") != identity:
                    raise ValueError("Timeline video identity mismatch")
                for field in ("actions", "goals", "why"):
                    ids = [row["id"] for row in video.get(field, [])]
                    if len(ids) != len(set(ids)):
                        raise ValueError(f"{identity}: duplicate {field} source IDs")
        else:
            self.warnings.append("Complete source asset unavailable. Annotation timelines show only known evaluation metadata and are partial.")
            self._fallback()
        self._queries = defaultdict(list)
        self._queries_by_id = {}
        for identity, reference in review_catalog.references.items():
            video_id = reference["video_id"]
            if video_id not in self.videos:
                self.videos[video_id] = self._empty_video(video_id)
            models = [model for model in review_catalog.models if identity in review_catalog.records[model]]
            query = {"id": identity, "task": reference["task"], "time_sec": reference["query_time_sec"],
                     "question": reference["question"], "models": models}
            video = self.videos[video_id]
            sidecar = review_catalog._media_records.get(identity, {})
            metadata = sidecar.get("scoring_metadata", {})
            action_id = (metadata.get("current_action") or {}).get("narration_id")
            if not action_id and reference["task"] == "explain" and ":explain:" in identity:
                action_id = identity.split(":explain:", 1)[1]
            known_action_ids = {action["id"] for action in video.get("actions", [])}
            if action_id not in known_action_ids:
                action_id = None
            time_sec = reference["query_time_sec"]
            active = [action["id"] for action in video.get("actions", [])
                      if action.get("start_sec") is not None and action.get("end_sec") is not None
                      and action["start_sec"] <= time_sec < action["end_sec"]]
            if action_id is None and reference["task"] == "track" and len(active) == 1:
                action_id = active[0]
            query["action_id"] = action_id
            query["active_action_ids"] = active
            query["goal_ids"] = [goal["id"] for goal in video.get("goals", [])
                                 if goal.get("start_sec") is not None and goal.get("end_sec") is not None
                                 and goal["start_sec"] <= time_sec < goal["end_sec"]]
            query["why_ids"] = [why["id"] for why in video.get("why", []) if why.get("action_id") == action_id]
            query["future_action_ids"] = [action["narration_id"] for action in metadata.get("future_actions", [])
                                           if action.get("narration_id") in known_action_ids]
            self._queries[video_id].append(query)
            self._queries_by_id[identity] = query
        for queries in self._queries.values():
            queries.sort(key=lambda row: (row["time_sec"], row["task"], row["id"]))

    def _empty_video(self, identity):
        entry = self._inventory.get(identity, {})
        return {"id": identity, "participant": entry.get("participant", identity.split("-")[0]),
                "partition": None, "duration_sec": entry.get("duration_sec"), "source_complete": False,
                "source_note": "Partial source metadata from evaluation queries; intervals outside those records are unavailable.",
                "actions": [], "goals": [], "why": [], "issues": []}

    def _fallback(self):
        identities = set(self._inventory) | {row["video_id"] for row in self.catalog.references.values()}
        self.videos = {identity: self._empty_video(identity) for identity in identities}
        gathered = {kind: defaultdict(dict) for kind in ("actions", "goals", "why")}
        for identity, reference in self.catalog.references.items():
            video_id = reference["video_id"]
            row = self.catalog._media_records.get(identity, {})
            metadata = row.get("scoring_metadata", {})
            video = self.videos[video_id]
            video["partition"] = row.get("partition", video.get("partition"))
            for action in ([metadata.get("current_action")] + metadata.get("history_actions", []) + metadata.get("future_actions", [])):
                if not action or not action.get("narration_id"):
                    continue
                pairs = action.get("main_action_class_ids") or []
                pair = pairs[0] if len(pairs) == 1 else []
                gathered["actions"][video_id][action["narration_id"]] = {
                    "id": action["narration_id"], "start_sec": action.get("start_sec"), "end_sec": action.get("end_sec"),
                    "text": action.get("action"), "verb_id": pair[0] if pair else None,
                    "noun_id": pair[1] if len(pair) > 1 else None, "class_pairs": pairs,
                    "verb": None, "noun": None, "narration": None, "valid_interval": True}
            goal = metadata.get("goal_annotation")
            if goal and goal.get("annotation_id"):
                gathered["goals"][video_id][goal["annotation_id"]] = {
                    "id": goal["annotation_id"], "start_sec": goal.get("start_sec"), "end_sec": goal.get("end_sec"),
                    "text": goal.get("label"), "scope": "as_annotated", "valid_interval": True}
            reason = metadata.get("reason_annotation")
            current = metadata.get("current_action") or {}
            if reason and current.get("narration_id"):
                why_id = current["narration_id"] + ":why"
                gathered["why"][video_id][why_id] = {
                    "id": why_id, "action_id": current["narration_id"], "start_sec": current.get("start_sec"),
                    "end_sec": current.get("end_sec"), "text": reason.get("reason_text"),
                    "status": reason.get("status"), "accepted": str(reason.get("status", "")).startswith("accepted"),
                    "narration": reason.get("original_narration"), "interval_meaning": "target_action_interval; not purpose duration",
                    "human_validated": False, "valid_interval": True}
        for video_id, video in self.videos.items():
            for kind in gathered:
                video[kind] = sorted(gathered[kind][video_id].values(), key=lambda row: (row.get("start_sec") or 0, row["id"]))

    def list_videos(self):
        result = []
        for identity, video in sorted(self.videos.items()):
            queries = self._queries.get(identity, [])
            task_counts = {task: sum(query["task"] == task for query in queries) for task in ("explain", "track")}
            result.append({"id": identity, "participant": video.get("participant"), "partition": video.get("partition"),
                           "duration_sec": video.get("duration_sec"), "action_count": len(video.get("actions", [])),
                           "goal_count": len(video.get("goals", [])), "why_count": len(video.get("why", [])),
                           "accepted_why_count": sum(row.get("accepted", False) for row in video.get("why", [])),
                           "query_count": len(queries), "query_task_counts": task_counts,
                           "query_tasks": [task for task, count in task_counts.items() if count],
                           "models": sorted({model for query in queries for model in query["models"]}),
                           "source_complete": video.get("source_complete", False)})
        return result

    def _video_media(self, identity):
        if identity in self._video_cache:
            return self._video_cache[identity]
        entry = self._inventory.get(identity, {})
        result = {"url": None, "available": False, "duration_sec": self.videos[identity].get("duration_sec")}
        if entry.get("path"):
            path = Path(entry["path"])
            if not path.is_absolute():
                path = self._inventory_path.parent / path
            if path.is_file() and self.catalog._video_matches(path, entry.get("fingerprint")):
                url = self.catalog._register(path, {".mp4", ".webm", ".mov", ".m4v"})
                result.update(url=url, available=bool(url))
            elif path.is_file():
                result["warning"] = "Video fingerprint mismatch; video is hidden."
            else:
                result["warning"] = "Source video file unavailable on this machine."
        elif identity in self.catalog._videos:
            result = dict(self.catalog._videos[identity])
        self._video_cache[identity] = result
        return result

    @staticmethod
    def _statistics(video):
        actions, goals, why = (video.get(kind, []) for kind in ("actions", "goals", "why"))
        accepted = [row for row in why if row.get("accepted")]
        duration = video.get("duration_sec")
        goal_groups = defaultdict(list)
        for goal in goals:
            goal_groups[goal.get("text")].append(goal)
        return {"action_count": len(actions), "goal_count": len(goals), "why_candidate_count": len(why),
                "accepted_why_count": len(accepted), "why_status_counts": dict(Counter(row.get("status") for row in why)),
                "action_coverage": _coverage(actions, duration), "goal_coverage": _coverage(goals, duration),
                "accepted_why_action_coverage": _coverage(accepted, duration),
                "why_coverage_interpretation": "Union of actions carrying an accepted narrated-purpose extraction; not duration of intent.",
                "verb_frequency": [{"id": identity, "count": count,
                                    "name": next((row.get("verb") for row in actions if row.get("verb_id") == identity), None)}
                                   for identity, count in Counter(row.get("verb_id") for row in actions if row.get("verb_id") is not None).most_common()],
                "noun_frequency": [{"id": identity, "count": count,
                                    "name": next((row.get("noun") for row in actions if row.get("noun_id") == identity), None)}
                                   for identity, count in Counter(row.get("noun_id") for row in actions if row.get("noun_id") is not None).most_common()],
                "goal_frequency": [{"text": text, "count": len(rows), **_coverage(rows, duration)}
                                   for text, rows in sorted(goal_groups.items(), key=lambda item: (-len(item[1]), item[0] or ""))],
                "missing_or_partial_action_classes": sum(row.get("verb_id") is None or row.get("noun_id") is None for row in actions)}

    def get_video(self, video_id):
        video = self.videos[video_id]
        media = self._video_media(video_id)
        issues = video.get("issues", [])
        warnings = list(self.warnings)
        counts = Counter(issue["category"] for issue in issues)
        warnings += [f"Source audit: {count} {kind.replace('_', ' ')}." for kind, count in counts.items()]
        if media.get("warning"):
            warnings.append(media["warning"])
        result = {key: video.get(key) for key in ("id", "participant", "partition", "duration_sec", "source_complete", "source_note")}
        result.update(actions=video.get("actions", []), goals=video.get("goals", []), why=video.get("why", []),
                      queries=self._queries.get(video_id, []), video=media, statistics=self._statistics(video), warnings=warnings,
                      source_issues=issues)
        result["statistics"]["query_count"] = len(result["queries"])
        return result

    def review_for_video(self, video_id, labels, model=None):
        """Use only validated saved human labels; never substitute automatic scores."""
        self.videos[video_id]
        points = []
        keys = set()
        for label in labels:
            query = self._queries_by_id.get(label.get("id"))
            if not query or self.catalog.references[label["id"]]["video_id"] != video_id:
                continue
            if label.get("model") not in query["models"] or model and label.get("model") != model:
                continue
            key = (label.get("reviewer"), label["id"], label["model"], label.get("field"))
            if key in keys:
                continue
            keys.add(key)
            points.append({"id": label["id"], "task": query["task"], "time_sec": query["time_sec"],
                           "model": label["model"], "field": label["field"], "label": label["label"],
                           "flags": label.get("flags", {}), "reviewer": label.get("reviewer"),
                           "verb_correct": label.get("verb_correct"), "noun_correct": label.get("noun_correct")})
        expected = 0
        for query in self._queries.get(video_id, []):
            reference = self.catalog.references[query["id"]]
            fields = int(reference["why"]["eligible"]) + int(reference["goal"]["eligible"])
            fields += int(reference["task"] == "track" and any(x is not None for x in reference["current_pair"]))
            fields += sum(any(x is not None for x in pair) for pair in reference["next_pairs"])
            expected += fields * sum(model is None or item == model for item in query["models"])
        reviewed = len({(point["id"], point["model"], point["field"]) for point in points})
        return {"summary": {"reviewed_fields": reviewed, "judgment_count": len(points), "expected_fields": expected,
                            "progress_fraction": reviewed / expected if expected else None,
                            "by_label": dict(Counter(point["label"] for point in points)),
                            "by_field": dict(Counter(point["field"] for point in points)),
                            "query_count": len(self._queries.get(video_id, []))},
                "points": sorted(points, key=lambda row: (row["time_sec"], row["model"], row["field"]))}
