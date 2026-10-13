"""Frozen validation parser, full denominators, NLL selection and review blinding."""
import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from code.e0.dataset import atomic_write_json, atomic_write_jsonl
from code.hdepic_experiment.metrics import aggregate_losses, blind_order, blind_review_ids, normalized, parse_prediction, protocol, score_predictions, select_checkpoint
from code.hdepic_experiment.report import build_report, write_blind_review


def record(identity, task="explain", goal=False):
    target = {"reason": "because it is hot"} if task == "explain" else {"current_action": "take bowl"}
    target["next_actions"] = ["take bowl", "wash bowl", "place bowl"]
    if goal:
        target["goal"] = "make soup"
    return {"id": identity, "task": task, "target": target, "partition": "validation", "participant": "P07", "video_id": "P07-video", "query_time_sec": 10., "input": {"history": [], "action_question_text": "pick pan"}, "scoring_metadata": {"reason_annotation": {"narration_id": identity, "char_start": 9, "char_end": 26, "original_narration": "pick pan because it is hot"}}}


def text(task="explain", goal=False):
    first = "<why>because it is hot</why>" if task == "explain" else "<action>take bowl</action>"
    return first + ("<goal>make soup</goal>" if goal else "") + "<next>take bowl; wash bowl; place bowl</next>"


def generation(identity, prediction, **extras):
    return {"id": identity, "prediction": prediction, "status": "completed", "finish_reason": "stop", "generated_tokens": 30, **extras}


def loss(identity, mean, tokens=10):
    return {"id": identity, "mean_nll": mean, "nll_sum": mean * tokens, "answer_tokens": tokens, "status": "completed"}


class ExperimentMetricsTests(unittest.TestCase):
    def test_strict_tag_parser_requested_fields_and_next_count(self):
        self.assertTrue(parse_prediction("\n " + text(goal=True) + " \n", "explain", True)["valid"])
        self.assertFalse(parse_prediction(text(goal=True), "explain", False)["valid"])
        self.assertFalse(parse_prediction(text(), "explain", True)["valid"])
        self.assertFalse(parse_prediction("Here is my answer: " + text(), "explain", False)["valid"])
        self.assertFalse(parse_prediction(text().replace("; place bowl", ""), "explain", False)["valid"])
        self.assertFalse(parse_prediction(text().replace("wash bowl", ""), "explain", False)["valid"])
        self.assertFalse(parse_prediction("<why>one</why>" + text(), "explain", False)["valid"])
        self.assertFalse(parse_prediction(text().replace("because it is hot", ""), "explain", False)["valid"])
        self.assertFalse(parse_prediction(None, "explain", False)["valid"])

    def test_normalization_is_lexical_not_semantic_and_decodes_entities_once(self):
        self.assertEqual(normalized("  ＴＡＫＥ\n Bowl  "), "take bowl")
        self.assertNotEqual(normalized("take bowl."), normalized("take bowl"))
        parsed = parse_prediction(text().replace("because it is hot", "hot &amp;amp; wet"), "explain", False)
        self.assertEqual(parsed["fields"]["why"], "hot &amp; wet")

    def test_missing_empty_malformed_and_length_keep_all_denominators(self):
        records = [record("e1", goal=True), record("e2"), record("t1", "track"), record("t2", "track", True)]
        rows = [generation("e1", text(goal=True), finish_reason="length"), generation("t1", "", generated_tokens=0), generation("t2", text("track", True).replace("; place bowl", ""))]
        result = score_predictions(records, rows)
        all_scope = result["scopes"]["all"]
        self.assertEqual(all_scope["expected"], 4)
        self.assertEqual(all_scope["strict_format_valid"], 1)
        self.assertEqual(all_scope["missing_prediction"], 1)
        self.assertEqual(all_scope["empty_output"], 2)
        self.assertEqual(all_scope["malformed_output"], 3)
        self.assertEqual(all_scope["finish_length"], 1)
        self.assertEqual(all_scope["zero_generated_tokens"], 1)
        self.assertEqual(all_scope["lexical_exact_match"]["why"], {"exact_matches": 1, "denominator": 2, "rate": .5})
        self.assertEqual(all_scope["lexical_exact_match"]["action"]["denominator"], 2)
        self.assertEqual(all_scope["lexical_exact_match"]["goal"]["denominator"], 2)
        self.assertEqual(all_scope["lexical_exact_match"]["next_sequence"]["denominator"], 4)
        for position in range(1, 4):
            self.assertEqual(all_scope["lexical_exact_match"][f"next_position_{position}"], {"exact_matches": 1, "denominator": 4, "rate": .25})

    def test_generation_error_never_scores_even_with_lexically_correct_text(self):
        result = score_predictions([record("e")], [generation("e", text(), status="error")])
        scope = result["scopes"]["all"]
        self.assertEqual(scope["generation_error"], 1)
        self.assertEqual(scope["strict_format_valid"], 0)
        self.assertEqual(scope["lexical_exact_match"]["why"]["rate"], 0)

    def test_partial_missing_empty_and_duplicate_field_diagnostics(self):
        records = [record("e1", goal=True), record("e2"), record("t1", "track")]
        rows = [generation("e1", "<why>reason</why><goal> </goal>"), generation("e2", "<why>one</why><why>two</why><next>unfinished"), generation("t1", "<next>take bowl; wash bowl; place bowl</next>")]
        summary = score_predictions(records, rows)["scopes"]["all"]
        d = summary["requested_field_diagnostics"]
        self.assertEqual(d["why"], {"denominator": 2, "missing": 0, "empty": 0, "duplicate": 1, "complete_nonempty": 1})
        self.assertEqual(d["goal"], {"denominator": 1, "missing": 0, "empty": 1, "duplicate": 0, "complete_nonempty": 0})
        self.assertEqual(d["action"]["missing"], 1)
        self.assertEqual(d["next"]["denominator"], 3)
        self.assertEqual(d["next"]["missing"], 2)
        self.assertEqual(summary["strict_format_valid"], 0)
        self.assertEqual(summary["lexical_exact_match"]["why"]["exact_matches"], 0)
        absent = score_predictions([record("missing", goal=True)], [])["scopes"]["all"]["requested_field_diagnostics"]
        self.assertEqual(absent["why"]["missing"], 1)
        self.assertEqual(absent["goal"]["missing"], 1)
        self.assertEqual(absent["next"]["missing"], 1)
        self.assertEqual(absent["action"]["denominator"], 0)

    def test_ordered_next_actions_and_positions_have_independent_scores(self):
        prediction = text().replace("take bowl; wash bowl; place bowl", "wash bowl; take bowl; place bowl")
        scope = score_predictions([record("e")], [generation("e", prediction)])["scopes"]["all"]["lexical_exact_match"]
        self.assertEqual(scope["next_sequence"]["rate"], 0)
        self.assertEqual(scope["next_position_1"]["rate"], 0)
        self.assertEqual(scope["next_position_2"]["rate"], 0)
        self.assertEqual(scope["next_position_3"]["rate"], 1)

    def test_duplicate_or_outside_prediction_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            score_predictions([record("e")], [generation("e", text()), generation("e", text())])
        with self.assertRaisesRegex(ValueError, "outside"):
            score_predictions([record("e")], [generation("other", text())])

    def test_nll_uses_example_mean_not_token_weighting_or_track_mean(self):
        records = [record("e1"), record("e2"), record("t", "track")]
        first = aggregate_losses(records, [loss("e1", 1., 1), loss("e2", 3., 99), loss("t", 1.)])
        second = aggregate_losses(records, [loss("e1", 1.9), loss("e2", 1.9), loss("t", 20.)])
        self.assertEqual(first["scopes"]["explain"]["mean_per_example_nll"], 2.)
        self.assertEqual(first["scopes"]["explain"]["token_weighted_nll"], 2.98)
        self.assertEqual(select_checkpoint({"base": first, "epoch1": first, "epoch2": second})["selected_stage"], "epoch2")
        self.assertEqual(select_checkpoint({"epoch1": first, "epoch2": first})["selected_stage"], "epoch1")

    def test_incomplete_nonfinite_or_zero_token_loss_cannot_select(self):
        records = [record("e"), record("t", "track")]
        complete = aggregate_losses(records, [loss("e", 1.), loss("t", 2.)])
        for bad_rows in ([loss("e", 1.)], [loss("e", float("nan")), loss("t", 2.)], [loss("e", 1., 0), loss("t", 2.)], [loss("e", 1.), {**loss("t", 2.), "nll_sum": 200.}]):
            incomplete = aggregate_losses(records, bad_rows)
            self.assertEqual(incomplete["scopes"]["all"]["expected_examples"], 2)
            self.assertIsNone(incomplete["scopes"]["all"]["mean_per_example_nll"])
            self.assertIsNone(select_checkpoint({"epoch1": complete, "epoch2": incomplete})["selected_stage"])

    def test_blind_sample_order_is_fixed_without_predictions(self):
        records = [record(f"e{i}") for i in range(100)]
        ids = blind_review_ids(records)
        self.assertEqual(len(ids), 50)
        self.assertEqual(ids, blind_review_ids(list(reversed(records))))
        self.assertEqual({blind_order(i) for i in ids}, {("base", "selected"), ("selected", "base")})
        self.assertEqual(protocol()["blind_review"]["size"], 50)

    def test_blind_csv_separate_key_preserves_reviewed_rows(self):
        records = [record(f"e{i}") for i in range(50)]
        ids = blind_review_ids(records)
        base = [generation(r["id"], text()) for r in records]
        selected = [generation(r["id"], text().replace("because it is hot", "to avoid burns")) for r in records]
        with tempfile.TemporaryDirectory() as name:
            run, data = Path(name) / "run", Path(name) / "data"
            atomic_write_jsonl(data / "swift/validation.jsonl", [{"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "prompt"}], "images": [f"/images/{i}.jpg" for i in range(16)]} for _ in records])
            path = write_blind_review(run, data, records, ids, base, selected, "epoch1")
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 50)
            self.assertNotIn("epoch1", rows[0])
            self.assertTrue(all(r["reviewer notes"] == "" for r in rows))
            self.assertEqual(len(json.loads(rows[0]["individual_images_json"])), 16)
            self.assertTrue((run / "review/blind_why_review_identity_key.json").exists())
            rows[0]["reviewer notes"] = "Independent reviewer annotation"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            write_blind_review(run, data, records, ids, base, selected, "epoch1")
            with path.open() as f:
                self.assertEqual(list(csv.DictReader(f))[0]["reviewer notes"], "Independent reviewer annotation")
            changed = [dict(r) for r in base]
            for row in changed:
                if row["id"] == ids[0]:
                    row["prediction"] = "changed output"
            with self.assertRaisesRegex(ValueError, "reviewed row"):
                write_blind_review(run, data, records, ids, changed, selected, "epoch1")

    def test_report_end_to_end_from_frozen_saved_validation_rows(self):
        records = [record(f"e{i}", goal=i % 2 == 0) for i in range(50)] + [record("track", "track")]
        with tempfile.TemporaryDirectory() as name:
            run, data = Path(name) / "run", Path(name) / "data"
            atomic_write_jsonl(data / "text/validation.jsonl", records)
            atomic_write_jsonl(data / "swift/validation.jsonl", [{"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "prompt"}], "images": [f"/images/{i}.jpg" for i in range(16)]} for _ in records])
            atomic_write_json(run / "frozen_evaluation.json", {"metric_protocol": protocol(), "blind_review_ids": blind_review_ids(records)})
            for stage, mean in (("base", 3.), ("epoch1", 2.), ("epoch2", 1.)):
                atomic_write_jsonl(run / f"predictions/{stage}/generation.jsonl", [generation(r["id"], text(r["task"], "goal" in r["target"])) for r in records])
                atomic_write_jsonl(run / f"predictions/{stage}/loss.jsonl", [loss(r["id"], mean) for r in records])
            with contextlib.redirect_stdout(io.StringIO()):
                output = build_report(run, data)
            self.assertEqual(output["checkpoint_selection"]["selected_stage"], "epoch2")
            self.assertEqual(output["stages"]["base"]["generation"]["scopes"]["all"]["lexical_exact_match"]["goal"]["denominator"], 25)
            self.assertTrue((run / "validation_report.md").exists())
            self.assertTrue((run / "review/blind_why_review.csv").exists())
            frozen = json.loads((run / "frozen_evaluation.json").read_text())
            frozen["metric_protocol"]["normalization"] = "changed after freeze"
            atomic_write_json(run / "frozen_evaluation.json", frozen)
            with self.assertRaisesRegex(ValueError, "frozen pre-baseline protocol"):
                build_report(run, data)


if __name__ == "__main__":
    unittest.main()
