"""Correction semantics: observed details, requested fields, and scope holds."""
import unittest

from code.hdepic_preparation.revise import HOLDS, QUESTION_OVERRIDES, question_text, render


def example(action, prefix, reason="because my hands are full", task="explain", goal=None, narration_id=None):
    target = {"reason": reason, "next_actions": ["take bowl", "wash bowl", "place bowl"]} if task == "explain" else {"current_action": action, "next_actions": ["take bowl", "wash bowl", "place bowl"]}
    if goal is not None:
        target["goal"] = goal
    r = {"task": task, "query_time_sec": 5., "input": {"history": []}, "target": target, "scoring_metadata": {"current_action": {"action": action}, "reason_annotation": {"original_narration": prefix + reason, "binding_clause": prefix, "char_start": len(prefix)}}}
    if narration_id:
        r["scoring_metadata"]["current_action"]["narration_id"] = narration_id
        r["scoring_metadata"]["reason_annotation"]["narration_id"] = narration_id
    if task == "explain":
        r["input"]["action_question_text"] = question_text(r)[0]
    return r


class RevisionTests(unittest.TestCase):
    def test_missing_goal_is_not_requested_or_targeted(self):
        frames = [{"timestamp_sec": i / 3, "path": f"/tmp/{i}.jpg"} for i in range(16)]
        for task in ("explain", "track"):
            for goal in (None, "make soup"):
                r = example("close drawer", "Close drawer ", task=task, goal=goal)
                item = render(r, frames)
                self.assertEqual(item["messages"][1]["content"].count("<goal>"), 0 if goal is None else 2)
                self.assertEqual("<goal>" in item["messages"][2]["content"], goal is not None)
                self.assertNotIn("when supervised", item["messages"][1]["content"])
                self.assertNotIn("stated reason", item["messages"][1]["content"])
                self.assertEqual(len(item["images"]), 16)
                if task == "track":
                    self.assertNotIn("close drawer", item["messages"][1]["content"])

    def test_hip_choice_kept_without_reason(self):
        r = example("close cutlery drawer", "Close the cutlery drawer with my hip ", narration_id="P05-20240425-124420-11")
        text, p = question_text(r)
        self.assertEqual(text, "close cutlery drawer (with my hip)")
        self.assertNotIn("hands are full", text)
        s = p["observed_modifiers"][0]
        self.assertEqual(r["scoring_metadata"]["reason_annotation"]["original_narration"][s["char_start"]:s["char_end"]], s["text"])
        self.assertEqual((s["char_start"], s["char_end"]), (25, 36))
        self.assertEqual(s["override_narration_id"], "P05-20240425-124420-11")
        self.assertEqual(s["attachment_check"], "explicit_source_text_review")

    def test_reviewed_quantity_and_careful_choice_without_embedded_purpose(self):
        r = example("pour squash", "Pour squash into the glass making sure that only a small amount is used ", "because it is quadruple strength", narration_id="P02-20240209-194141-455")
        self.assertIn("only a small amount", question_text(r)[0])
        r = example("tilt bowl", "Tilt the bowl on top of the sink to drop the water in it, tilting only carefully ", "because only water needs to be dropped", narration_id="P01-20240204-142301-40")
        self.assertIn("only carefully", question_text(r)[0])
        self.assertNotIn("drop the water", question_text(r)[0])

    def test_reviewed_destination_and_glove_choice(self):
        r = example("put colander", "put the colander on the worktop instead ", "because it does not fit", narration_id="P07-20240529-191007-380")
        self.assertIn("on the worktop instead", question_text(r)[0])
        r = example("wear glove", "wear the glove using the right hand. Notice that this is the wrong glove direction so I'm wearing what should be the left hand glove in the right hand ", "because the other is wet", narration_id="P01-20240204-152537-15")
        self.assertIn("left hand glove in the right hand", question_text(r)[0])

    def test_broad_hand_tool_location_enrichment_is_disabled(self):
        r = example("pull door", "I pull the door with my pinky ", "because the other fingers are dirty")
        self.assertEqual(question_text(r)[0], "pull door")
        r = example("pick up pan lid", "pick up the pan lid with my right hand using the oven glove ", "because it is hot")
        self.assertEqual(question_text(r)[0], "pick up pan lid")
        r = example("put colander", "put the colander on the worktop instead ", "because it does not fit")
        self.assertEqual(question_text(r)[0], "put colander")
        r = example("close cutlery drawer", "Close the cutlery drawer with my hip ")
        self.assertEqual(question_text(r)[0], "close cutlery drawer")

    def test_regression_left_hand_holds_glass_not_processor_bowl(self):
        r = example("tilt food processor bowl", " While holding the glass using the left hand, tilt the food processor bowl, which is in the right hand, ", "so that the juice inside it would be poured into the glass that's being held with the left hand.", narration_id="P01-20240202-110250-210")
        text, provenance = question_text(r)
        self.assertEqual(text, "tilt food processor bowl")
        self.assertEqual(provenance["observed_modifiers"], [])

    def test_regression_left_hand_holds_pan_not_stirring_spatula(self):
        r = example("stir potatoes", "while holding the frying pan with my left hand using the spatula that i'm holding in my right hand stir the potatoes ", "so that the salt is evenly distributed", narration_id="P03-20240216-185832-669")
        text, provenance = question_text(r)
        self.assertEqual(text, "stir potatoes")
        self.assertEqual(provenance["observed_modifiers"], [])

    def test_regression_spoon_is_on_lid_not_directly_on_countertop(self):
        r = example("place spoon", " place the spoon on top of the lid that is on the countertop ", "so that no spots are on the surface. So any dirt that is on the spoon will fall onto the lid as opposed to the surface.", narration_id="P01-20240202-195538-90")
        text, provenance = question_text(r)
        self.assertEqual(text, "place spoon")
        self.assertEqual(provenance["observed_modifiers"], [])

    def test_reviewed_override_fails_closed_on_changed_source_or_action(self):
        with self.assertRaisesRegex(AssertionError, "evidence"):
            example("close cutlery drawer", "Close the cutlery drawer with my hand ", narration_id="P05-20240425-124420-11")
        with self.assertRaisesRegex(AssertionError, "action"):
            example("close oven", "Close the cutlery drawer with my hip ", narration_id="P05-20240425-124420-11")
        self.assertTrue(all(not n.startswith(("P08", "P09")) for n in QUESTION_OVERRIDES))

    def test_source_conflict_and_uncertain_scope_held_without_test_overrides(self):
        self.assertIn("P04-20240415-181212-279", HOLDS)
        self.assertIn("P02-20240211-120927-670", HOLDS)
        self.assertTrue(all(not n.startswith(("P08", "P09")) for n in HOLDS))


if __name__ == "__main__":
    unittest.main()
