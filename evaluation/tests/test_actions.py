"""Ordered forecasting semantics, unknown preservation, and class eligibility."""
import unittest

from hdepic_eval.actions import (
    damerau_osa_distance, levenshtein_distance, score_action_pairs, score_next,
)


class ActionMetricTests(unittest.TestCase):
    def test_gold_component_eligibility_and_unknown_predictions(self):
        scores = score_action_pairs(
            [(1, 7), (None, 8), (9, None), (None, None)],
            [(1, 7), (2, 8), (9, None), (None, 4)],
        )
        self.assertEqual(scores["verb"], {
            "correct": 2, "denominator": 3, "rate": 2 / 3, "mapped_predictions": 2})
        self.assertEqual(scores["noun"], {
            "correct": 2, "denominator": 3, "rate": 2 / 3, "mapped_predictions": 2})
        self.assertEqual(scores["action"], {
            "correct": 1, "denominator": 2, "rate": .5, "mapped_predictions": 1})

    def test_ordered_positions_distinguish_shift_from_recovery(self):
        reference = [(1, 11), (2, 12), (3, 13)]
        scores = score_next([[(2, 12), (3, 13), (4, 14)]], [reference])
        self.assertEqual(scores["mean_position_accuracy"]["action"], 0)
        self.assertEqual(scores["exact_ordered_sequence"]["action"]["rate"], 0)
        # Delete the last prediction and insert the first gold action: two
        # Levenshtein operations despite three position errors.
        self.assertAlmostEqual(scores["normalized_edit_distance"]["action"]["mean"], 2 / 3)

    def test_adjacent_swap_costs_two_not_a_transposition(self):
        self.assertEqual(levenshtein_distance([1, 2, 3], [2, 1, 3]), 2)
        scores = score_next([[(2, 12), (1, 11), (3, 13)]],
                            [[(1, 11), (2, 12), (3, 13)]])
        self.assertAlmostEqual(scores["mean_position_accuracy"]["action"], 1 / 3)
        self.assertAlmostEqual(scores["normalized_edit_distance"]["action"]["mean"], 2 / 3)
        self.assertAlmostEqual(scores["normalized_damerau_osa"]["action"]["mean"], 1 / 3)

    def test_restricted_damerau_is_not_unrestricted_damerau(self):
        # Unrestricted Damerau can transpose CA to AC, then insert B: cost 2.
        # OSA forbids editing the same substring twice and yields cost 3.
        self.assertEqual(damerau_osa_distance("CA", "ABC"), 3)
        self.assertEqual(damerau_osa_distance("ABC", "CA"), 3)
        self.assertEqual(damerau_osa_distance([], []), 0)
        self.assertEqual(damerau_osa_distance([1, 2], []), 2)
        self.assertEqual(damerau_osa_distance([1, 2, 3], [2, 1, 3]), 1)

    def test_unknown_slot_is_not_dropped_or_shifted(self):
        reference = [(1, 11), (2, 12), (3, 13)]
        scores = score_next([[(1, 11), (None, None), (3, 13)]], [reference])
        self.assertEqual(scores["position_2"]["action"]["rate"], 0)
        self.assertEqual(scores["position_3"]["action"]["rate"], 1)
        self.assertAlmostEqual(scores["normalized_edit_distance"]["action"]["mean"], 1 / 3)
        self.assertEqual(scores["forecast_diagnostics"]["incomplete"], 0)

    def test_short_prediction_pads_and_extras_are_reported_and_truncated(self):
        references = [[(1, 11), (2, 12), (3, 13)]] * 2
        scores = score_next([[(1, 11)], [(1, 11), (2, 12), (3, 13), (4, 14)]], references)
        self.assertEqual(scores["position_1"]["action"]["rate"], 1)
        self.assertEqual(scores["position_2"]["action"]["rate"], .5)
        self.assertEqual(scores["position_3"]["action"]["rate"], .5)
        self.assertEqual(scores["exact_ordered_sequence"]["action"]["correct"], 1)
        self.assertAlmostEqual(scores["normalized_edit_distance"]["action"]["mean"], 1 / 3)
        self.assertEqual(scores["forecast_diagnostics"], {
            "examples": 2, "horizon": 3, "extracted_count": 5,
            "incomplete": 1, "extra": 1, "missing_positions": 2, "extra_actions": 1,
        })

    def test_partial_gold_keeps_verb_sequence_but_excludes_noun_and_action(self):
        scores = score_next([[(1, 11), (2, 12), (3, 13)]],
                            [[(1, 11), (2, 12), (3, None)]])
        self.assertEqual(scores["position_3"]["verb"]["denominator"], 1)
        self.assertEqual(scores["position_3"]["noun"]["denominator"], 0)
        self.assertIsNone(scores["position_3"]["action"]["rate"])
        self.assertEqual(scores["exact_ordered_sequence"]["verb"]["rate"], 1)
        self.assertEqual(scores["exact_ordered_sequence"]["noun"]["denominator"], 0)
        self.assertIsNone(scores["normalized_edit_distance"]["action"]["mean"])
        self.assertEqual(scores["mean_position_accuracy"]["action"], 1)

    def test_mean_position_accuracy_is_macro_over_eligible_position_rates(self):
        scores = score_next(
            [[(1, 11), (2, 12)], [(1, 11), (None, None)]],
            [[(1, 11), (2, 12)], [(7, 17), (None, None)]],
            horizon=2,
        )
        # Position rates are 1/2 and 1/1. Pooling three gold positions would
        # yield 2/3, but the specified mean position metric yields 3/4.
        self.assertEqual(scores["mean_position_accuracy"]["action"], .75)
        self.assertEqual(scores["normalized_edit_distance"]["action"]["denominator"], 1)

    def test_all_missing_output_fails_complete_reference(self):
        scores = score_next([[]], [[(1, 11), (2, 12), (3, 13)]])
        self.assertEqual(scores["exact_ordered_sequence"]["verb"]["denominator"], 1)
        self.assertEqual(scores["exact_ordered_sequence"]["verb"]["rate"], 0)
        self.assertEqual(scores["normalized_edit_distance"]["action"]["mean"], 1)
        self.assertEqual(scores["forecast_diagnostics"]["missing_positions"], 3)

    def test_empty_population_has_no_synthetic_successes(self):
        scores = score_next([], [])
        self.assertIsNone(scores["mean_position_accuracy"]["action"])
        self.assertIsNone(scores["exact_ordered_sequence"]["verb"]["rate"])
        self.assertIsNone(scores["normalized_edit_distance"]["noun"]["mean"])
        self.assertEqual(scores["forecast_diagnostics"]["examples"], 0)

    def test_class_zero_is_a_real_class_and_noun_errors_do_not_erase_verbs(self):
        scores = score_action_pairs([(0, 2)], [(0, 3)])
        self.assertEqual(scores["verb"]["rate"], 1)
        self.assertEqual(scores["noun"]["rate"], 0)
        self.assertEqual(scores["action"]["rate"], 0)

    def test_short_gold_excludes_only_unavailable_positions(self):
        scores = score_next([[(1, 11), (2, 12), (3, 13)]], [[(1, 11)]])
        self.assertEqual(scores["position_1"]["action"]["rate"], 1)
        self.assertIsNone(scores["position_2"]["action"]["rate"])
        self.assertIsNone(scores["normalized_edit_distance"]["action"]["mean"])

    def test_invalid_class_ids_and_shapes_fail_instead_of_becoming_unknown(self):
        for invalid in ((True, 1), (-1, 2), (1.0, 2), (float("nan"), 2),
                        (float("inf"), 2), ("1", 2), (1,), "12", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    score_action_pairs([invalid], [(1, 2)])
        with self.assertRaises(ValueError):
            score_action_pairs([], [(1, 2)])
        with self.assertRaises(ValueError):
            score_next([[]], [])
        for invalid_horizon in (0, -1, True, 3.0, None):
            with self.subTest(horizon=invalid_horizon):
                with self.assertRaises(ValueError):
                    score_next([], [], horizon=invalid_horizon)


if __name__ == "__main__":
    unittest.main()
