"""Frozen reference binding, shared semantic masks, and causal metadata checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from hdepic_eval.references import canonical_pair, load_references, load_reference_bundle


def action(identity="n0", start=8.0, end=10.0, pair=None):
    return {"narration_id": identity, "video_id": "P07-video", "action": "move mug",
            "start_sec": start, "end_sec": end,
            "main_action_class_ids": [[11, 13]] if pair is None else [pair],
            "pair_class_ids": [[99, 99], [88, 88]]}


def reference(task="explain"):
    narration = "Move the mug so that it fits in the cupboard and wipe the surface."
    reason = "so that it fits in the cupboard and wipe the surface"
    start = narration.index(reason)
    row = {
        "id": f"P07-video:{task}:n0", "task": task, "video_id": "P07-video",
        "participant": "P07", "query_time_sec": 10.0,
        "current_action": action(end=10.0 if task == "explain" else 11.0),
        "history_actions": [action("h1", 1.0, 2.0)],
        "future_actions": [action("f1", 11.0, 12.0), action("f2", 13.0, 14.0),
                           action("f3", 15.0, 16.0)],
        "frames": [{"timestamp_sec": 5.0}, {"timestamp_sec": 10.0}],
        "goal_status": "available",
        "goal_annotation": {"annotation_id": "activities.csv:2",
                            "resolved_video_id": "P07-video",
                            "label": "Unloading the dishwasher",
                            "start_sec": 0.0, "end_sec": 20.0},
        "reason_annotation": None,
    }
    if task == "explain":
        row["reason_annotation"] = {
            "narration_id": "n0", "video_id": "P07-video",
            "reason_text": reason, "original_narration": narration,
            "char_start": start, "char_end": start + len(reason),
            "status": "accepted_pending_human_review",
        }
        row["question_provenance"] = {"action_question_text": "move blue mug"}
    return row


def legacy(row, status="valid_with_nonpurpose_suffix", model="base", core=None):
    return {"id": row["id"], "model": model, "reference_status": status,
            "purpose_core": core if core is not None else "so that it fits in the cupboard",
            "reference_reason": row["reason_annotation"]["reason_text"],
            "label": "incorrect"}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def load(self, records, reviews=None):
        path = self.root / "validation.scoring.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        review_path = None
        if reviews is not None:
            review_path = self.root / "legacy.jsonl"
            review_path.write_text("".join(json.dumps(row) + "\n" for row in reviews))
        return load_references(path, review_path)

    def test_original_annotations_and_scope_are_preserved(self):
        row = reference()
        prepared = self.load([row])[0]
        self.assertEqual(prepared["question"], "move blue mug")
        self.assertEqual(prepared["current_pair"], [11, 13])
        self.assertEqual(prepared["next_pairs"], [[11, 13]] * 3)
        self.assertTrue(prepared["why"]["eligible"])
        self.assertEqual(prepared["why"]["source_status"], "annotation_derived_pending_review")
        self.assertEqual(prepared["why"]["reference"], row["reason_annotation"]["reason_text"])
        self.assertEqual(prepared["goal"]["reference"], "Unloading the dishwasher")
        self.assertEqual(prepared["goal"]["profile"]["scope_level"], "as_annotated")
        self.assertEqual(prepared["goal"]["profile"]["source_annotation_ids"], ["activities.csv:2"])
        self.assertEqual(prepared["history"][0]["end_sec"], 2.0)

    def test_frozen_bundle_loads_without_raw_dataset_or_review_files(self):
        prepared = self.load([reference()])
        path = self.root / 'references.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in prepared))
        self.assertEqual(load_reference_bundle(path), prepared)
        prepared[0]['why']['reference'] = 'unbound reason'
        path.write_text(json.dumps(prepared[0]) + '\n')
        with self.assertRaises(ValueError):
            load_reference_bundle(path)

    def test_purpose_core_comes_from_consistent_frozen_source_not_answer_label(self):
        row = reference()
        first = legacy(row)
        second = legacy(row, model="epoch1")
        second["label"] = "correct"
        prepared = self.load([row], [first, second])[0]["why"]
        self.assertEqual(prepared["reference"], "so that it fits in the cupboard")
        self.assertEqual(prepared["original_reference"], row["reason_annotation"]["reason_text"])
        self.assertEqual(prepared["source_status"], "valid_with_nonpurpose_suffix")
        self.assertIn("legacy_frozen_why_review:", prepared["source_provenance"])

    def test_inconsistent_common_source_mask_or_core_is_rejected(self):
        row = reference()
        for change in ({"reference_status": "ambiguous_attachment"},
                       {"purpose_core": "so that it fits"}):
            with self.subTest(change=change):
                second = {**legacy(row, model="epoch1"), **change}
                with self.assertRaisesRegex(ValueError, "inconsistent legacy WHY"):
                    self.load([row], [legacy(row), second])

    def test_ambiguous_source_is_excluded_for_every_model(self):
        row = reference()
        reviews = [legacy(row, status="ambiguous_attachment", model=model)
                   for model in ("base", "epoch1")]
        for item in reviews:
            item["purpose_core"] = None
        why = self.load([row], reviews)[0]["why"]
        self.assertFalse(why["eligible"])
        self.assertIsNone(why["reference"])
        self.assertEqual(why["original_reference"], row["reason_annotation"]["reason_text"])

    def test_stale_reason_or_unbound_purpose_core_cannot_be_reused(self):
        row = reference()
        for change, message in (({"reference_reason": "different reason"}, "differs from"),
                                ({"purpose_core": "so that dinner is ready"}, "bound source substring")):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, message):
                    self.load([row], [{**legacy(row), **change}])

    def test_reason_offsets_and_action_binding_are_checked(self):
        for change in ({"char_start": 0}, {"narration_id": "another-action"},
                       {"video_id": "another-video"}):
            row = reference()
            row["reason_annotation"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load([row])

    def test_legacy_action_question_and_narration_binding(self):
        row = reference()
        for change in ({'action_question': 'another question'}, {'narration_id': 'another narration'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load([row], [{**legacy(row), **change}])

    def test_partial_canonical_class_preserves_verb_and_ignores_alternatives(self):
        row = reference()
        row["future_actions"][2]["main_action_class_ids"] = [[15]]
        self.assertEqual(self.load([row])[0]["next_pairs"][2], [15, None])
        self.assertEqual(canonical_pair({"main_action_class_ids": []}), [None, None])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            canonical_pair({"main_action_class_ids": [[1, 2], [3, 4]]})
        with self.assertRaises(ValueError):
            canonical_pair({"main_action_class_ids": [[True, 2]]})

    def test_absent_or_ambiguous_goal_is_unsupervised_not_a_negative(self):
        for status in ("missing_activity", "ambiguous_activity"):
            row = reference()
            row["goal_status"] = status
            row["goal_annotation"] = None
            goal = self.load([row])[0]["goal"]
            self.assertFalse(goal["eligible"])
            self.assertIsNone(goal["reference"])
            self.assertIsNone(goal["profile"])
            self.assertEqual(goal["source_status"], status)

    def test_goal_interval_is_half_open_and_video_binding_is_checked(self):
        for change in ({"end_sec": 10.0}, {"start_sec": 10.01},
                       {"resolved_video_id": "another-video"}):
            row = reference()
            row["goal_annotation"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load([row])
        row = reference()
        row["goal_annotation"]["start_sec"] = 10.0
        self.assertTrue(self.load([row])[0]["goal"]["eligible"])

    def test_track_has_no_why_and_uses_current_action_question(self):
        row = reference("track")
        prepared = self.load([row])[0]
        self.assertFalse(prepared["why"]["eligible"])
        self.assertEqual(prepared["why"]["source_status"], "not_applicable")
        self.assertEqual(prepared["question"], "move mug")
        row["query_time_sec"] = 11.0
        with self.assertRaisesRegex(ValueError, "outside the current action"):
            self.load([row])

    def test_future_history_and_frames_cannot_leak_past_query_boundary(self):
        mutations = [
            ("future_at_query", lambda r: r["future_actions"][0].update(start_sec=10.0)),
            ("future_other_video", lambda r: r["future_actions"][0].update(video_id="other")),
            ("wrong_order", lambda r: r["future_actions"].reverse()),
            ("short_horizon", lambda r: r["future_actions"].pop()),
            ("unfinished_history", lambda r: r["history_actions"][0].update(end_sec=10.1)),
            ("future_frame", lambda r: r["frames"][0].update(timestamp_sec=10.1)),
            ("wrong_explain_end", lambda r: r.update(query_time_sec=9.0)),
        ]
        for name, mutate in mutations:
            row = reference()
            mutate(row)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load([row])

    def test_duplicate_ids_and_unknown_tasks_are_rejected(self):
        row = reference()
        with self.assertRaisesRegex(ValueError, "Duplicate reference id"):
            self.load([row, copy.deepcopy(row)])
        row["task"] = "other"
        with self.assertRaisesRegex(ValueError, "unsupported task"):
            self.load([row])


if __name__ == "__main__":
    unittest.main()
