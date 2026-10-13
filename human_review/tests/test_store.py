import json
import tempfile
import unittest
from pathlib import Path

from store import ConflictError, LabelStore, export_labels, progress


class FixtureCatalog:
    models = ["model_a", "model_b"]

    def __init__(self):
        self.hash = "reference-and-answer-hash"
        self.references = {"example": {"task": "track", "why": {"eligible": True},
            "goal": {"eligible": True}, "current_pair": [1, 2],
            "next_pairs": [[1, 2], [3, 4], [5, None]]}}
        self.records = {m: {"example": {}} for m in self.models}

    def get_example(self, identity):
        if identity != "example":
            raise KeyError(identity)
        return {"reference": {"why": {"eligible": True}, "goal": {"eligible": True},
                              "current": {"eligible": True, "pair": [1, 2]},
                              "next": [{"pair": [1, 2]}, {"pair": [3, 4]}, {"pair": [5, None]}]},
                "predictions": [{"model": m, "answer_sha256": self.hash,
                    "semantic_packets": {"why": {"packet_id": "why-packet", "prompt_sha256": "why-hash"},
                                         "goal": {"packet_id": "goal-packet", "prompt_sha256": "goal-hash"}}}
                    for m in self.models]}

    def bootstrap(self):
        return {"title": "Test Review", "models": [{"id": m, "name": m} for m in self.models]}

    def list_examples(self):
        return [{"id": "example", "task": "track"}]


def payload(**updates):
    result = {"reviewer": "Alice", "id": "example", "model": "model_a",
              "answer_sha256": "reference-and-answer-hash", "field": "why", "label": "correct",
              "notes": "The answer states the annotated purpose.", "flags": {}, "evidence": "observed"}
    result.update(updates)
    return result


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "labels.sqlite3"
        self.catalog = FixtureCatalog()
        self.store = LabelStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_persistent_label_and_audit_on_revision(self):
        first = self.store.save(payload(), self.catalog, 0)
        second = self.store.save(payload(label="partial"), self.catalog, 1)
        self.assertEqual((first["revision"], second["revision"]), (1, 2))
        reloaded = LabelStore(self.path)
        self.assertEqual(reloaded.all("Alice")[0]["label"], "partial")
        with reloaded.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)

    def test_stale_revision_does_not_overwrite(self):
        self.store.save(payload(), self.catalog, 0)
        with self.assertRaises(ConflictError):
            self.store.save(payload(label="incorrect"), self.catalog, 0)
        self.assertEqual(self.store.all()[0]["label"], "correct")

    def test_changed_source_excludes_old_labels_and_rejects_stale_write(self):
        self.store.save(payload(), self.catalog)
        self.catalog.hash = "new-reference-hash"
        self.assertEqual(self.store.current(self.catalog), [])
        self.assertEqual(len(self.store.all()), 1)
        with self.assertRaises(ConflictError):
            self.store.save(payload(), self.catalog)

    def test_reviewers_have_independent_labels(self):
        self.store.save(payload(), self.catalog)
        self.store.save(payload(reviewer="Bob", label="incorrect"), self.catalog)
        self.assertEqual(self.store.all("Alice")[0]["label"], "correct")
        self.assertEqual(self.store.all("Bob")[0]["label"], "incorrect")

    def test_uncertain_and_reference_issues_stay_in_raw_export(self):
        self.store.save(payload(label="uncertain"), self.catalog)
        self.store.save(payload(field="goal", label="partial", flags={"reference_issue": True}), self.catalog)
        labels = self.store.current(self.catalog)
        self.assertEqual(len(export_labels(labels, "jsonl")[2].splitlines()), 2)
        self.assertEqual(export_labels(labels, "judgments")[2], "")

    def test_semantic_export_is_evaluator_contract(self):
        label = self.store.save(payload(flags={"unsupported_addition": False}), self.catalog)
        _, _, output = export_labels([label], "judgments")
        row = json.loads(output)
        self.assertEqual(row["packet_id"], "why-packet")
        self.assertEqual(row["prompt_sha256"], "why-hash")
        self.assertEqual(row["judge"], "human:Alice")
        self.assertEqual(row["judgment"]["flags"], {"unsupported_addition": False})
        self.assertEqual(row["assessment_status"], "human_review")

    def test_duplicate_semantic_packets_deduplicated_and_conflicts_rejected(self):
        a = self.store.save(payload(), self.catalog)
        b = self.store.save(payload(model="model_b", notes="A second equivalent explanation."), self.catalog)
        self.assertEqual(len(export_labels([a, b], "judgments")[2].splitlines()), 1)
        b = self.store.save(payload(model="model_b", label="incorrect"), self.catalog)
        with self.assertRaises(ConflictError):
            export_labels([a, b], "judgments")

    def test_component_consistency_and_missing_gold(self):
        with self.assertRaises(ValueError):
            self.store.save(payload(field="next_1", verb_correct=False), self.catalog)
        with self.assertRaises(ValueError):
            self.store.save(payload(field="next_3", noun_correct=True), self.catalog)
        with self.assertRaises(ValueError):
            self.store.save(payload(field="next_3", label="correct", verb_correct=True), self.catalog)
        saved = self.store.save(payload(field="next_3", label="uncertain", verb_correct=True), self.catalog)
        self.assertIsNone(saved["noun_correct"])
        saved = self.store.save(payload(field="next_3", label="uncertain", verb_correct=False), self.catalog)
        self.assertIs(saved["verb_correct"], False)

    def test_reference_ineligible_rejected(self):
        original = self.catalog.get_example
        def excluded(identity):
            detail = original(identity)
            detail["reference"]["why"]["eligible"] = False
            return detail
        self.catalog.get_example = excluded
        with self.assertRaises(ValueError):
            self.store.save(payload(), self.catalog)

    def test_invalid_flags_and_central_contradiction_rejected(self):
        for flags in ({"reference_issue": "yes"}, {"new": True}, {"contradiction": True}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.store.save(payload(flags=flags), self.catalog)

    def test_csv_formula_text_escaped_json_unmodified(self):
        label = self.store.save(payload(notes="=SUM(A1:A2)"), self.catalog)
        self.assertIn("'=SUM(A1:A2)", export_labels([label], "csv")[2])
        self.assertEqual(json.loads(export_labels([label], "jsonl")[2])["notes"], "=SUM(A1:A2)")

    def test_progress_separates_partial_labeling_and_uncertainty(self):
        label = self.store.save(payload(label="uncertain"), self.catalog)
        report = progress(self.catalog, [label])
        self.assertEqual(report["reviewed"], 0)
        self.assertEqual(report["total_predictions"], 2)
        self.assertTrue(report["examples"]["example"]["uncertain"])
        self.assertEqual(report["labels"], 1)


if __name__ == "__main__":
    unittest.main()
