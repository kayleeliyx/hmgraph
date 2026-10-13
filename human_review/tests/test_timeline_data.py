"""Full source timelines preserve scope, overlaps, provenance and sparse reviews."""
import csv
import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("timeline_builder", "build_timeline.py")
runtime = load("timeline_runtime", "timeline_data.py")


class TimelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reason_path = self.root / "reason_labels.jsonl"
        self.inventory_path = self.root / "video_inventory.json"
        self.activities = self.root / "activities"
        self.classes = self.root / "classes"
        self.output = self.root / "timeline.json.gz"
        self.activities.mkdir()
        self.classes.mkdir()
        for kind, values in (("verb", [(0, "take"), (1, "put")]), ("noun", [(1, "cup")])):
            with (self.classes / f"HD_EPIC_{kind}_classes.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "key", "instances"])
                for identity, key in values:
                    writer.writerow([identity, key, repr([key])])
        self.other_video = "P09-20240623-154547"
        self.video_path = self.root / "v.mp4"
        self.video_path.write_bytes(b"video")
        self.inventory = {"v": {"video_id": "v", "participant": "P01", "duration_sec": 20.0, "path": "v.mp4"},
                          self.other_video: {"video_id": self.other_video, "participant": "P09", "duration_sec": 30.0}}
        self.inventory_path.write_text(json.dumps(self.inventory))
        rows = [self.reason("v-1", "v", 1, 3, "take cup", [[0, 1]], "rejected"),
                self.reason("v-2", "v", 2, 4, "put cup", [[1, 1], [0, 1]], "accepted_pending_human_review", "because hungry"),
                self.reason("v-3", "v", 8, 9, "take cup", [[0, 1]], "quarantined_pending_human_review", "because unsure"),
                self.reason("other-1", self.other_video, 4, 5, "put cup", [[1, 1]], "rejected", "because unrelated")]
        self.reason_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with (self.activities / "P01_recipe_timestamps.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_id", "recipe_id", "high_level_activity_label", "start_time", "end_time"])
            writer.writerow(["v", "", "Cook lunch", 0, 10])
            writer.writerow(["v", "", "Wash dishes", 5, 15])
            writer.writerow(["v", "", "Preserved reversed source", 12, 11])
            writer.writerow(["P09-2024623-154547", "", "Finish cooking", 0, "end"])
        self.asset = builder.build(self.reason_path, self.inventory_path, self.activities, self.classes, self.output)
        reference = {"id": "v:track:2.500", "task": "track", "video_id": "v", "query_time_sec": 2.5,
                     "question": "take cup", "why": {"eligible": False}, "goal": {"eligible": True},
                     "current_pair": [0, 1], "next_pairs": [[1, 1], [0, 1], [1, None]]}
        self.media = {}

        def register(path, kinds):
            identity = hashlib.sha256(str(path).encode()).hexdigest()
            self.media[identity] = path
            return "/media/" + identity

        self.catalog = SimpleNamespace(
            config={"timeline_annotations": "timeline.json.gz", "timeline_video_inventory": "video_inventory.json"},
            _path=lambda value: (self.root / value).resolve(), references={reference["id"]: reference},
            models=["model"], records={"model": {reference["id"]: {}}}, _media_records={}, _videos={},
            _video_matches=Mock(return_value=True), _register=Mock(side_effect=register))

    @staticmethod
    def reason(identity, video_id, start, end, text, pairs, status, reason=None):
        narration = text + (" " + reason if reason else "") + "."
        return {"narration_id": identity, "video_id": video_id, "participant": "P09" if video_id.startswith("P09") else "P01",
                "partition": "test" if video_id.startswith("P09") else "initial_sft", "status": status,
                "original_narration": narration, "reason_text": reason,
                "char_start": narration.find(reason) if reason else None,
                "char_end": narration.find(reason) + len(reason) if reason else None,
                "target": {"action": text, "start_sec": start, "end_sec": end,
                           "main_action_class_ids": pairs, "main_actions": [[text.split()[0], "cup"]],
                           "class_label_status": "available"}}

    def test_builder_retains_all_source_videos_and_overlaps(self):
        self.assertEqual(self.asset["audit"]["retained_actions"], 4)
        self.assertEqual(self.asset["audit"]["retained_goals"], 4)
        self.assertEqual(self.asset["audit"]["video_count"], 2)
        self.assertEqual(self.asset["videos"]["v"]["actions"][0]["start_sec"], 1)
        self.assertEqual(self.asset["videos"]["v"]["actions"][1]["start_sec"], 2)
        self.assertEqual(self.asset["videos"]["v"]["goals"][1]["start_sec"], 5)
        self.assertTrue(self.asset["videos"][self.other_video]["source_complete"])

    def test_builder_preserves_original_scope_and_invalid_interval(self):
        goals = self.asset["videos"]["v"]["goals"]
        self.assertEqual(goals[0]["text"], "Cook lunch")
        reversed_goal = next(row for row in goals if row["text"] == "Preserved reversed source")
        self.assertEqual((reversed_goal["start_sec"], reversed_goal["end_sec"]), (12.0, 11.0))
        self.assertFalse(reversed_goal["valid_interval"])
        self.assertEqual(self.asset["audit"]["issue_counts"]["reversed_activity_interval"], 1)
        corrected = self.asset["videos"][self.other_video]["goals"][0]
        self.assertEqual(corrected["original_video_id"], "P09-2024623-154547")
        self.assertEqual(corrected["resolved_video_id"], self.other_video)
        self.assertEqual(corrected["end_sec"], 30.0)
        self.assertEqual(corrected["original_end_time"], "end")

    def test_builder_distinguishes_why_candidates_from_accepted_source(self):
        self.assertEqual(self.asset["audit"]["why_candidates"], 3)
        self.assertEqual(self.asset["audit"]["accepted_why"], 1)
        why = self.asset["videos"]["v"]["why"]
        self.assertEqual([(row["status"], row["accepted"]) for row in why],
                         [("accepted_pending_human_review", True), ("quarantined_pending_human_review", False)])
        self.assertEqual((why[0]["start_sec"], why[0]["end_sec"]), (2.0, 4.0))
        self.assertIn("not purpose duration", why[0]["interval_meaning"])
        self.assertFalse(why[0]["human_validated"])

    def test_multiple_class_pairs_are_preserved_without_picking_one(self):
        action = self.asset["videos"]["v"]["actions"][1]
        self.assertEqual(action["class_pairs"], [[1, 1], [0, 1]])
        self.assertIsNone(action["verb_id"])
        self.assertIsNone(action["noun_id"])

    def test_asset_is_deterministic_and_source_hashes_are_real(self):
        before = self.output.read_bytes()
        builder.build(self.reason_path, self.inventory_path, self.activities, self.classes, self.output)
        self.assertEqual(before, self.output.read_bytes())
        for path, expected in self.asset["provenance"]["source_sha256"].items():
            self.assertEqual(builder.sha256(path), expected)

    def test_catalog_all_videos_have_counts_but_only_saved_queries_have_models(self):
        timeline = runtime.TimelineCatalog(self.catalog)
        videos = {row["id"]: row for row in timeline.list_videos()}
        self.assertEqual(videos["v"]["query_count"], 1)
        self.assertEqual(videos["v"]["query_task_counts"], {"explain": 0, "track": 1})
        self.assertEqual(videos["v"]["query_tasks"], ["track"])
        self.assertEqual(videos["v"]["models"], ["model"])
        self.assertEqual(videos[self.other_video]["query_count"], 0)
        self.assertEqual(videos[self.other_video]["query_tasks"], [])
        self.assertEqual(videos[self.other_video]["models"], [])
        self.assertEqual(videos[self.other_video]["action_count"], 1)
        self.assertEqual(timeline.get_video("v")["queries"][0]["goal_ids"], ["P01_recipe_timestamps.csv:2"])

    def test_union_coverage_differs_from_annotation_count_and_keeps_gaps(self):
        timeline = runtime.TimelineCatalog(self.catalog)
        stats = timeline.get_video("v")["statistics"]
        self.assertEqual(stats["action_count"], 3)
        self.assertEqual(stats["action_coverage"]["union_duration_sec"], 4.0)
        self.assertEqual(stats["goal_coverage"]["union_duration_sec"], 15.0)
        self.assertEqual(stats["goal_coverage"]["uncovered_duration_sec"], 5.0)
        self.assertEqual(stats["goal_coverage"]["excluded_invalid_intervals"], 1)
        self.assertEqual(stats["accepted_why_action_coverage"]["union_duration_sec"], 2.0)

    def test_media_is_lazily_registered_and_fingerprint_checked(self):
        timeline = runtime.TimelineCatalog(self.catalog)
        self.catalog._video_matches.assert_not_called()
        timeline.list_videos()
        self.catalog._video_matches.assert_not_called()
        media = timeline.get_video("v")["video"]
        self.assertTrue(media["available"])
        self.catalog._video_matches.assert_called_once()
        self.assertEqual(self.media[media["url"].split("/")[-1]], self.video_path)

    def test_failed_video_fingerprint_hides_media(self):
        self.catalog._video_matches.return_value = False
        video = runtime.TimelineCatalog(self.catalog).get_video("v")
        self.assertFalse(video["video"]["available"])
        self.assertIn("fingerprint mismatch", " ".join(video["warnings"]))

    def test_saved_human_labels_are_points_and_auto_scores_are_not_invented(self):
        timeline = runtime.TimelineCatalog(self.catalog)
        self.assertEqual(timeline.review_for_video("v", [])["points"], [])
        labels = [{"id": "v:track:2.500", "model": "model", "reviewer": "A", "field": "next_1", "label": "correct", "flags": {}},
                  {"id": "v:track:2.500", "model": "model", "reviewer": "B", "field": "next_1", "label": "incorrect", "flags": {}}]
        result = timeline.review_for_video("v", labels)
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["summary"]["reviewed_fields"], 1)
        self.assertEqual(result["summary"]["expected_fields"], 5)
        self.assertEqual(result["summary"]["progress_fraction"], 0.2)
        self.assertTrue(all(point["time_sec"] == 2.5 for point in result["points"]))
        self.assertTrue(all("end_sec" not in point for point in result["points"]))
        self.assertEqual(timeline.review_for_video(self.other_video, labels)["points"], [])

    def test_missing_source_asset_falls_back_visibly_partial(self):
        self.catalog.config["timeline_annotations"] = "unavailable.gz"
        timeline = runtime.TimelineCatalog(self.catalog)
        self.assertEqual(len(timeline.list_videos()), 2)
        self.assertFalse(timeline.get_video("v")["source_complete"])
        self.assertIn("partial", " ".join(timeline.get_video("v")["warnings"]).lower())
        self.assertEqual(timeline.get_video("v")["actions"], [])

    def test_duplicate_source_actions_and_wrong_asset_version_are_rejected(self):
        self.asset["videos"]["v"]["actions"].append(self.asset["videos"]["v"]["actions"][0])
        with gzip.open(self.output, "wt", encoding="utf-8") as handle:
            json.dump(self.asset, handle)
        with self.assertRaisesRegex(ValueError, "duplicate actions"):
            runtime.TimelineCatalog(self.catalog)
        self.asset["version"] = "changed"
        with gzip.open(self.output, "wt", encoding="utf-8") as handle:
            json.dump(self.asset, handle)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            runtime.TimelineCatalog(self.catalog)


if __name__ == "__main__":
    unittest.main()
