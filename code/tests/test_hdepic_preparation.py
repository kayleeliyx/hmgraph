"""Protocol boundary tests; no models, network, or dataset mutation."""
import unittest

from code.hdepic_preparation.build import action_metadata, extract_reason, frame_plan
from code.hdepic_preparation.export import render


def annotation(narration, action=("open", "drawer"), pairs=None):
    return {"unique_narration_id": "P01-video-1", "video_id": "P01-video", "participant_id": "P01", "narration": narration, "main_actions": [action], "pairs": pairs or [action], "main_action_classes": [(3, 3)], "pair_classes": [(3, 3)], "start_timestamp": 1., "end_timestamp": 2.}


class PreparationTests(unittest.TestCase):
    def test_reason_exact_span_and_word_overlap_allowed(self):
        text = "Open the drawer so that I can put utensils in the drawer."
        reason = extract_reason(annotation(text))
        self.assertEqual(reason["status"], "accepted_pending_human_review")
        self.assertEqual(text[reason["char_start"]:reason["char_end"]], "so that I can put utensils in the drawer")

    def test_reason_bound_to_other_action_is_rejected(self):
        reason = extract_reason(annotation("Open the drawer and lift the spoon so that I can wash it.", pairs=[("open", "drawer"), ("lift", "spoon")]))
        self.assertEqual(reason["rejection_category"], "intervening_other_action")

    def test_unrelated_event_because_is_rejected(self):
        reason = extract_reason(annotation("Pick up the potato chunk. The potato chunk will burn because it is small.", ("pick up", "potato chunk")))
        self.assertEqual(reason["status"], "rejected")

    def test_generic_to_does_not_invent_reason(self):
        reason = extract_reason(annotation("Move the drawer to the countertop.", ("move", "drawer")))
        self.assertEqual(reason["rejection_category"], "no_explicit_marker")

    def test_cannot_and_incidental_observation_are_not_purpose(self):
        for text in ["Open the drawer but it is stuck so I can't open it.", "Open the drawer. So you can hear the sound of its contents."]:
            self.assertEqual(extract_reason(annotation(text))["status"], "rejected")

    def test_because_about_failed_result_not_target_action(self):
        reason = extract_reason(annotation("Open the drawer, but when it won't open because the spoon is in the way, move it."))
        self.assertEqual(reason["rejection_category"], "because_explains_failure_or_other_event")

    def test_missing_noun_id_is_preserved_as_missing(self):
        row = annotation("Roll.", ("roll", ""))
        row["main_action_classes"] = [(47,)]
        meta = action_metadata(row)
        self.assertEqual(meta["verb_ids"], [47])
        self.assertEqual(meta["noun_ids"], [None])
        self.assertEqual(meta["class_label_status"], "missing_noun_class")

    def test_fractional_endpoint_never_requests_future_frame(self):
        from pathlib import Path
        plan = frame_plan({"frame_count": 1000, "video_id": "vid", "cache_id": "cache"}, 5.001, Path("/tmp/hdepic-test"))
        self.assertEqual(len(plan), 16)
        self.assertEqual(plan[-1]["planned_timestamp_sec"], 5.)
        self.assertTrue(all(.001 <= f["planned_timestamp_sec"] <= 5.001 for f in plan))
        self.assertEqual(len({f["source_frame_index"] for f in plan}), 16)
        plan = frame_plan({"frame_count": 1000, "video_id": "vid", "cache_id": "cache"}, 8.4, Path("/tmp/hdepic-test"))
        self.assertEqual(plan[0]["planned_timestamp_sec"], 3.4)
        self.assertEqual(plan[-1]["planned_timestamp_sec"], 8.4)

    def test_track_omits_current_action_input_and_missing_goal_target(self):
        row = {"task": "track", "query_time_sec": 5., "input": {"history": []}, "target": {"current_action": "open drawer", "next_actions": ["take spoon", "close drawer", "wash spoon"]}}
        frames = [{"timestamp_sec": i / 3, "path": f"/tmp/{i}.jpg"} for i in range(16)]
        result = render(row, frames)
        self.assertNotIn("open drawer", result["messages"][1]["content"])
        self.assertNotIn("<goal>", result["messages"][2]["content"])
        self.assertEqual(result["messages"][1]["content"].count("<image>"), 16)


if __name__ == "__main__":
    unittest.main()
