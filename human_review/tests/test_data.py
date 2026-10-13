"""Portable data bindings: no model calls, annotation installation or GPU."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location("review_data", Path(__file__).resolve().parents[1] / "data.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Catalog, semantic_packet = _MODULE.Catalog, _MODULE.semantic_packet


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "saved"
        self.ref = {"id": "v:track:10", "task": "track", "video_id": "v", "participant": "P",
                    "query_time_sec": 10.0, "question": "wash cup", "history": [],
                    "current_pair": [1, 1], "next_pairs": [[0, 1], [1, 2], [1, None]],
                    "why": {"eligible": False, "reference": None, "original_reference": None,
                            "source_status": "not_applicable", "original_narration": None},
                    "goal": {"eligible": True, "reference": "Wash dishes", "source_status": "available",
                             "profile": {"original_label": "Wash dishes", "required_units": ["Wash dishes"]}}}
        self.raw = "<action>wash cup</action><goal>Clean dishes</goal><next>take cup;wash bowl</next>"
        self.rec = {"id": self.ref["id"], "task": "track", "video_id": "v", "participant": "P",
                    "current_pair": self.ref["current_pair"], "next_pairs": self.ref["next_pairs"],
                    "generation_completed": True, "raw_prediction_sha256": hashlib.sha256(self.raw.encode()).hexdigest(),
                    "fields": {"action": {"text": "wash cup"}, "why": {"text": None}, "goal": {"text": "Clean dishes"}},
                    "predicted_current": [1, 1], "predicted_next": [[0, 1], [1, None]],
                    "next_extraction": {"actions": ["take cup", "wash bowl"]},
                    "semantic": {"why": {"eligible": False, "prediction": None, "judgment": None},
                                 "goal": {"eligible": True, "prediction": "Clean dishes", "judgment": None,
                                          "packet_id": semantic_packet(self.ref, "goal", "Clean dishes")["packet_id"]}}}
        self.conf = {"evaluation_dir": "../saved", "evaluation_config": "../eval_config.json",
                     "media_records": "../dataset/media.jsonl", "video_inventory": "../dataset/videos.json",
                     "title": "Fixture Review"}
        self.config_path = self.root / "review" / "config.json"
        write_json(self.config_path, self.conf)
        write_json(self.root / "eval_config.json", {"models": [{"name": "model", "predictions": "predictions.jsonl"}]})
        write_rows(self.root / "predictions.jsonl", [{"id": self.ref["id"], "prediction": self.raw}])
        write_json(self.run / "mapper.json", {"verb_map": {"wash": 1, "take": 0},
                                              "noun_map": {"cup": 1, "bowl": 2}, "provenance": {}})
        self.frame_path = self.root / "dataset" / "frame.jpg"
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_path.write_bytes(b"fixture-image")
        self.video_path = self.root / "dataset" / "video.mp4"
        self.video_path.write_bytes(b"fixture-video")
        self.sidecar = {"id": self.ref["id"], "task": "track", "video_id": "v", "query_time_sec": 10.0,
                        "frames": [{"path": "frame.jpg", "index": 0, "planned_timestamp_sec": 5.0}],
                        "scoring_metadata": {"current_action": {"action": "wash cup", "main_action_class_ids": [[1, 1]],
                                                                 "start_sec": 9.0, "end_sec": 11.0},
                                             "future_actions": [{"action": "take cup", "main_action_class_ids": [[0, 1]], "start_sec": 12.0, "end_sec": 13.0},
                                                                {"action": "wash bowl", "main_action_class_ids": [[1, 2]], "start_sec": 14.0, "end_sec": 15.0},
                                                                {"action": "wash", "main_action_class_ids": [[1]], "start_sec": 16.0, "end_sec": 17.0}],
                                             "goal_status": "available", "goal_annotation": {"label": "Wash dishes"}}}
        write_rows(self.root / "dataset" / "media.jsonl", [self.sidecar])
        write_json(self.root / "dataset" / "videos.json", {"v": {"path": "video.mp4", "duration_sec": 30.0}})
        self.persist()

    def persist(self):
        write_rows(self.run / "references.jsonl", [self.ref])
        write_rows(self.run / "per_example" / "model.jsonl", [self.rec])

    def test_portable_catalog_preserves_component_eligibility_and_missing_slots(self):
        catalog = Catalog(self.config_path)
        boot = catalog.bootstrap()
        self.assertEqual(boot["example_count"], 1)
        self.assertEqual(boot["title"], "Fixture Review")
        self.assertEqual(boot["tasks"], ["track"])
        example = catalog.get_example(self.ref["id"])
        prediction = example["predictions"][0]
        self.assertEqual(prediction["fields"]["next"], ["take cup", "wash bowl", None])
        self.assertEqual(prediction["auto"]["current"], {"verb": True, "noun": True, "pair": True})
        self.assertEqual(prediction["auto"]["next"][1], {"verb": True, "noun": False, "pair": False})
        self.assertEqual(prediction["auto"]["next"][2], {"verb": False, "noun": None, "pair": None})
        self.assertEqual(prediction["raw"], self.raw)
        self.assertTrue(prediction["raw_available"])
        self.assertTrue(example["media"]["video"]["available"])
        self.assertEqual(example["media"]["future_end_sec"], 17.0)
        frame = example["media"]["frames"][0]
        self.assertEqual(catalog.media[frame["url"].split("/")[-1]], self.frame_path)
        self.assertEqual(example["reference"]["next"][0]["class_names_source"]["verb"], "mapper_alias_representative")
        self.assertTrue(catalog.list_examples()[0]["mapping_error"])
        self.assertTrue(catalog.list_examples()[0]["semantic_pending"])

    def test_raw_hash_mismatch_hides_changed_prediction(self):
        write_rows(self.root / "predictions.jsonl", [{"id": self.ref["id"], "prediction": "changed"}])
        catalog = Catalog(self.config_path)
        self.assertIsNone(catalog.get_example(self.ref["id"])["predictions"][0]["raw"])
        self.assertIn("hash/status", " ".join(catalog.warnings))

    def test_failed_raw_prediction_never_exposed(self):
        write_rows(self.root / "predictions.jsonl", [{"id": self.ref["id"], "prediction": self.raw, "status": "failed"}])
        catalog = Catalog(self.config_path)
        self.assertFalse(catalog.get_example(self.ref["id"])["predictions"][0]["raw_available"])

    def test_changed_reference_invalidates_review_identity(self):
        before = Catalog(self.config_path).get_example(self.ref["id"])["predictions"][0]["answer_sha256"]
        self.ref["query_time_sec"] = 11.0
        self.persist()
        after = Catalog(self.config_path).get_example(self.ref["id"])["predictions"][0]["answer_sha256"]
        self.assertNotEqual(before, after)

    def test_importing_review_decision_does_not_invalidate_identity(self):
        before = Catalog(self.config_path).get_example(self.ref["id"])["predictions"][0]
        self.rec["semantic"]["goal"]["judgment"] = {
            "field": "goal", "label": "correct", "justification": "Same activity."}
        self.persist()
        after = Catalog(self.config_path).get_example(self.ref["id"])["predictions"][0]
        self.assertEqual(before["answer_sha256"], after["answer_sha256"])
        self.assertNotEqual(before["record_sha256"], after["record_sha256"])

    def test_changed_mapping_invalidates_review_identity(self):
        before = Catalog(self.config_path).get_example(self.ref["id"])["predictions"][0]["answer_sha256"]
        self.rec["predicted_current"] = [0, 1]
        self.persist()
        after = Catalog(self.config_path).get_example(self.ref["id"])["predictions"][0]["answer_sha256"]
        self.assertNotEqual(before, after)

    def test_media_mismatch_suppresses_wrong_sidecar(self):
        self.sidecar["scoring_metadata"]["goal_annotation"]["label"] = "Prepare coffee"
        write_rows(self.root / "dataset" / "media.jsonl", [self.sidecar])
        catalog = Catalog(self.config_path)
        self.assertEqual(catalog.get_example(self.ref["id"])["media"]["frames"], [])
        self.assertIn("disagree", " ".join(catalog.warnings))

    def test_future_frame_is_not_shown_as_observation(self):
        self.sidecar["frames"][0]["planned_timestamp_sec"] = 12.0
        write_rows(self.root / "dataset" / "media.jsonl", [self.sidecar])
        self.assertEqual(Catalog(self.config_path).get_example(self.ref["id"])["media"]["frames"], [])

    def test_duplicate_and_out_of_cohort_records_rejected(self):
        write_rows(self.run / "per_example" / "model.jsonl", [self.rec, self.rec])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            Catalog(self.config_path)
        other = copy.deepcopy(self.rec)
        other["id"] = "outside"
        write_rows(self.run / "per_example" / "model.jsonl", [other])
        with self.assertRaisesRegex(ValueError, "out-of-cohort"):
            Catalog(self.config_path)

    def test_semantic_contract_mismatch_is_rejected(self):
        self.rec["semantic"]["goal"]["packet_id"] = "changed"
        self.persist()
        with self.assertRaisesRegex(ValueError, "semantic packet differs"):
            Catalog(self.config_path)

    def test_packet_reconstructed_for_already_judged_answers(self):
        self.rec["semantic"]["goal"]["judgment"] = {"field": "goal", "label": "correct"}
        self.persist()
        example = Catalog(self.config_path).get_example(self.ref["id"])
        self.assertEqual(example["predictions"][0]["semantic_packets"]["goal"], semantic_packet(self.ref, "goal", "Clean dishes"))

    def test_optional_media_and_raw_are_not_required(self):
        write_json(self.config_path, {"evaluation_dir": "../saved"})
        catalog = Catalog(self.config_path)
        example = catalog.get_example(self.ref["id"])
        self.assertIsNone(example["predictions"][0]["raw"])
        self.assertFalse(example["media"]["video"]["available"])
        self.assertEqual(example["reference"]["next"][0]["text"], "take cup")

    def test_video_sampled_fingerprint_change_is_hidden(self):
        stat = self.video_path.stat()
        hasher = hashlib.sha256()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime_ns).encode())
        hasher.update(self.video_path.read_bytes())
        fingerprint = {"method": "sha256(size,mtime_ns,first_1MiB,last_1MiB)", "value": hasher.hexdigest(),
                       "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        write_json(self.root / "dataset" / "videos.json", {"v": {"path": "video.mp4", "fingerprint": fingerprint}})
        self.assertTrue(Catalog(self.config_path).get_example(self.ref["id"])["media"]["video"]["available"])
        self.video_path.write_bytes(b"changed-video")
        catalog = Catalog(self.config_path)
        self.assertFalse(catalog.get_example(self.ref["id"])["media"]["video"]["available"])
        self.assertIn("fingerprint mismatch", " ".join(catalog.warnings))


if __name__ == "__main__":
    unittest.main()
