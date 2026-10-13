import unittest

from server import video_review_points


class QueryCatalog:
    models = ["a", "b"]

    def __init__(self):
        self.references = {
            "explain": {"video_id": "video", "task": "explain", "query_time_sec": 8.0,
                        "why": {"eligible": True}, "goal": {"eligible": True},
                        "current_pair": [1, 2], "next_pairs": [[1, 2], [3, 4], [5, None]]},
            "track": {"video_id": "video", "task": "track", "query_time_sec": 10.0,
                      "why": {"eligible": False}, "goal": {"eligible": True},
                      "current_pair": [1, 2], "next_pairs": [[1, 2], [3, 4], [5, 6]]},
            "elsewhere": {"video_id": "other", "task": "explain", "query_time_sec": 2.0,
                          "why": {"eligible": True}, "goal": {"eligible": True},
                          "current_pair": [1, 2], "next_pairs": [[1, 2], [3, 4], [5, 6]]},
        }
        self.records = {"a": {key: {} for key in self.references}, "b": {"explain": {}}}


class TimelineQueryTests(unittest.TestCase):
    def test_saved_and_unreviewed_are_query_points(self):
        labels = [{"id": "explain", "model": "a", "field": "why", "label": "partial",
                   "flags": {}, "notes": "A true part of the reference purpose."}]
        summary = video_review_points(QueryCatalog(), "video", labels)
        self.assertEqual(summary["eligible_count"], 15)
        self.assertEqual(summary["counts"]["unreviewed"], 14)
        self.assertEqual(summary["counts"]["partial"], 1)
        self.assertTrue(all("start_sec" not in row and "end_sec" not in row for row in summary["points"]))
        self.assertEqual({row["time_sec"] for row in summary["points"]}, {8.0, 10.0})

    def test_model_filter_and_missing_model_cohort(self):
        summary = video_review_points(QueryCatalog(), "video", [], "b")
        self.assertEqual(summary["eligible_count"], 5)
        self.assertEqual({p["id"] for p in summary["points"]}, {"explain"})
        with self.assertRaises(ValueError):
            video_review_points(QueryCatalog(), "video", [], "unknown")

    def test_no_evaluation_queries_has_zero_points_not_zero_accuracy(self):
        summary = video_review_points(QueryCatalog(), "training_video", [])
        self.assertEqual(summary["eligible_count"], 0)
        self.assertEqual(summary["points"], [])
        self.assertNotIn("accuracy", summary)

    def test_excluded_why_and_track_current_eligibility(self):
        summary = video_review_points(QueryCatalog(), "video", [])
        track_fields = {p["field"] for p in summary["points"] if p["id"] == "track"}
        explain_fields = {p["field"] for p in summary["points"] if p["id"] == "explain"}
        self.assertNotIn("why", track_fields)
        self.assertNotIn("current", explain_fields)
        self.assertIn("current", track_fields)


if __name__ == "__main__":
    unittest.main()
