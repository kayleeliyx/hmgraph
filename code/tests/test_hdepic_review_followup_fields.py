"""Offline supplementary field boundaries, denominators and repetition."""
import unittest

from code.hdepic_experiment.metrics import normalized, parse_prediction
from code.hdepic_review_followup.fields import extract_field, score_fields
from code.hdepic_review_followup.forecast import parse_triple, repetition_rows, repetition_summary


def record(identity, task="explain", goal=True):
    target = {"reason": "because it is hot"} if task == "explain" else {"current_action": "take bowl"}
    target["next_actions"] = ["take bowl", "wash bowl", "place bowl"]
    if goal:
        target["goal"] = "make soup"
    return {"id": identity, "task": task, "target": target}


def prediction(identity, text, **extra):
    return {"id": identity, "prediction": text, "status": "completed", **extra}


class SupplementaryFieldTests(unittest.TestCase):
    def test_valid_why_and_goal_survive_malformed_next(self):
        text = "<why>because it is hot</why><goal>make soup</goal><next>take bowl\nwash bowl\nplace bowl</next>"
        self.assertFalse(parse_prediction(text, "explain", True)["valid"])
        result = score_fields([record("e")], [prediction("e", text)])
        self.assertEqual(result["scopes"]["all"]["why"]["exact_matches"], 1)
        self.assertEqual(result["scopes"]["all"]["goal"]["exact_matches"], 1)
        self.assertEqual(result["original_strict_metric"]["scopes"]["all"]["lexical_exact_match"]["why"]["exact_matches"], 0)
        self.assertEqual(result["per_example"][0]["raw_prediction"], text)

    def test_duplicate_why_rejected_without_choosing_reference_match(self):
        text = "<why>different</why><why>because it is hot</why>"
        field = extract_field(text, "why")
        self.assertTrue(field["duplicate"])
        self.assertTrue(field["ambiguous"])
        self.assertFalse(field["extractable"])
        self.assertIsNone(field["text"])
        self.assertEqual(score_fields([record("e")], [prediction("e", text)])["scopes"]["all"]["why"]["exact_matches"], 0)

    def test_missing_closing_and_additional_unmatched_opening_are_not_guessed(self):
        absent_close = extract_field("<why>because it is hot", "why")
        self.assertTrue(absent_close["missing"])
        self.assertTrue(absent_close["ambiguous"])
        extra_open = extract_field("<why>because it is hot</why><why>unfinished", "why")
        self.assertTrue(extra_open["duplicate"])
        self.assertFalse(extra_open["extractable"])

    def test_nested_crossed_and_reversed_boundaries_are_ambiguous(self):
        for text in ("<why><goal>purpose</goal></why>", "<why>purpose<goal>activity</why></goal>", "</why>purpose<why>"):
            with self.subTest(text=text):
                item = extract_field(text, "why")
                self.assertTrue(item["ambiguous"])
                self.assertFalse(item["extractable"])

    def test_malformed_different_field_and_surrounding_prose_do_not_block_pair(self):
        field = extract_field("prose<next><why>because it is hot</why>more prose", "why")
        self.assertTrue(field["extractable"])
        self.assertEqual(field["text"], "because it is hot")

    def test_single_html_unescape_original_normalization_and_exact_offsets(self):
        raw = "prefix<why>  ＨＯＴ &amp;amp; wet  </why>suffix"
        field = extract_field(raw, "why")
        self.assertEqual(field["text"], "ＨＯＴ &amp; wet")
        self.assertEqual(normalized(field["text"]), "hot &amp; wet")
        self.assertEqual(raw[field["raw_char_start"]:field["raw_char_end"]], field["raw_field_text"])
        self.assertTrue(extract_field("<why>&nbsp;</why>", "why")["empty"])
        self.assertEqual(extract_field("<WHY>reason</WHY>", "why")["status"], "missing")

    def test_fixed_task_goal_denominators_missing_and_error_rows(self):
        records = [record("e1"), record("e2", goal=False), record("t1", "track"), record("t2", "track", False)]
        rows = [prediction("e1", "<why>because it is hot</why><goal>make soup</goal>"), prediction("t1", "<action>take bowl</action><goal>make soup</goal>", status="error"), prediction("t2", "<action>take bowl</action>")]
        result = score_fields(records, rows)
        all_fields = result["scopes"]["all"]
        self.assertEqual(all_fields["why"]["denominator"], 2)
        self.assertEqual(all_fields["action"]["denominator"], 2)
        self.assertEqual(all_fields["goal"]["denominator"], 2)
        self.assertEqual(all_fields["why"]["exact_matches"], 1)
        self.assertEqual(all_fields["action"]["exact_matches"], 1)
        self.assertEqual(all_fields["goal"]["exact_matches"], 1)
        self.assertEqual(all_fields["why"]["missing_prediction"], 1)
        self.assertEqual(all_fields["action"]["generation_error"], 1)
        self.assertEqual(len(result["per_example"]), 4)

    def test_outside_and_duplicate_ids_fail(self):
        for rows in ([prediction("outside", "")], [prediction("e", ""), prediction("e", "")]):
            with self.assertRaises(ValueError):
                score_fields([record("e")], rows)

    def test_forecast_normalizes_identical_strings_without_whole_format_gate(self):
        parsed = parse_triple("prose<why>unfinished<next>ＴＡＫＥ bowl; take   bowl; Take bowl</next>")
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["normalized_actions"], ["take bowl"] * 3)
        records = [record("repeat"), record("bad"), record("missing")]
        rows = repetition_rows(records, [prediction("repeat", "<next>ＴＡＫＥ bowl; take   bowl; Take bowl</next>"), prediction("bad", "<next>take bowl; place bowl</next>")])
        summary = repetition_summary(rows)
        self.assertEqual(summary["denominator"], 3)
        self.assertEqual(summary["all_three_identical"], 1)
        self.assertEqual(summary["malformed_triples"], 2)
        self.assertEqual([r["id"] for r in rows if r["all_three_identical"]], ["repeat"])

    def test_forecast_preserves_reference_normalization_without_html_decode(self):
        r = record("r")
        r["target"]["next_actions"] = [" ＰＵＴ bowl ", "put bowl", "PUT bowl"]
        rows = repetition_rows([r])
        self.assertTrue(rows[0]["all_three_identical"])
        self.assertEqual(rows[0]["normalized_reference_triple"], ["put bowl"] * 3)
        parsed = parse_triple("<next>a &amp; b; a &amp; b; a &amp; b</next>")
        self.assertEqual(parsed["normalized_actions"], ["a & b"] * 3)
        # The frozen parser decodes before semicolon splitting. Retain that
        # rule even when a doubly encoded entity retains a literal semicolon.
        nested_entity = parse_triple("<next>a &amp;amp; b; a &amp;amp; b; a &amp;amp; b</next>")
        self.assertEqual(nested_entity["extraction"]["text"], "a &amp; b; a &amp; b; a &amp; b")
        self.assertFalse(nested_entity["valid"])
        self.assertFalse(parse_triple("<next>a; a; a</next><next>b; b; b</next>")["valid"])


if __name__ == "__main__":
    unittest.main()
