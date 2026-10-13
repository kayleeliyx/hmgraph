"""CPU-only workflow completion/cache guards; all data are temporary fixtures."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from code.hdepic_sft_qwen38 import checkpoints, workflow
from code.hdepic_sft_qwen38.common import canonical, read_json, sha, write_json

IDENTITY = {"config_sha256": "c" * 64, "model_revision": "fixture_revision",
            "data_sha256": {"initial_sft": "d" * 64, "validation": "v" * 64}, "world_size": 2}


class WorkflowCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.scoring = [{"id": f"validation-fixture-{i:04d}",
                         "task": "explain" if i < 228 else "track"} for i in range(2209)]
        train = types.ModuleType("code.hdepic_sft_qwen38.train")
        train.identity = mock.Mock(return_value=IDENTITY)
        data = types.ModuleType("code.hdepic_sft_qwen38.data")
        data.load_partition = mock.Mock(return_value=([], self.scoring, []))
        self.partition_loader = data.load_partition
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(sys.modules, {train.__name__: train, data.__name__: data}).start()
        mock.patch.object(workflow, "RUN", self.run).start()
        self.gpu_probe = mock.patch.object(workflow, "gpu_probe", side_effect=AssertionError("GPU probe forbidden in CPU test")).start()
        self.popen = mock.patch.object(workflow.subprocess, "Popen", side_effect=AssertionError("Job launch forbidden in CPU test")).start()

    def tearDown(self):
        self.gpu_probe.assert_not_called()
        self.popen.assert_not_called()

    def cache(self, stage="epoch1"):
        folder = self.run / "validation" / stage
        folder.mkdir(parents=True, exist_ok=True)
        records = [{**record, "row_index": i, "nll": 0.5 + i / 10000,
                    "supervised_tokens": 30, "stage": stage} for i, record in enumerate(self.scoring)]
        loss = folder / "loss.jsonl"
        loss.write_text("".join(json.dumps(r) + "\n" for r in records))
        write_json(folder / "completed.json", {"identity": IDENTITY, "stage": stage,
                   "records": 2209, "loss_sha256": sha(loss)})
        return folder, records

    def rehash_cache(self, folder, records):
        loss = folder / "loss.jsonl"
        loss.write_text("".join(json.dumps(r) + "\n" for r in records))
        marker = read_json(folder / "completed.json")
        marker["loss_sha256"] = sha(loss)
        write_json(folder / "completed.json", marker)

    def checkpoint(self, step=1818):
        """Inventory fixture; real DTensor serialization has its own Gloo tests."""
        path = self.run / "training/main/checkpoints" / f"checkpoint-{step}"
        path.mkdir(parents=True)
        for name in checkpoints.RANK_FILES + checkpoints.ADAPTER_FILES:
            (path / name).write_bytes((name + str(step)).encode())
        write_json(path / "completed.json", {"format": checkpoints.FORMAT, "passed": True,
                   "world_size": 2, "identity": IDENTITY,
                   "identity_sha256": hashlib.sha256(canonical(IDENTITY)).hexdigest(),
                   "state": {"step": step, "epoch": 1 if step == 1818 else 0,
                             "microbatch_consumed": 7272, "sampler_state": {"seed": 20261003}},
                   "trainable_schema": {"fixture.lora_A.default.weight": {"shape": [2, 4]}},
                   "files": {name: checkpoints._file(path / name)
                             for name in checkpoints.RANK_FILES + checkpoints.ADAPTER_FILES}})
        write_json(self.run / "training/main/completed.json", {"identity": IDENTITY,
                   "passed": True, "optimizer_steps": 1818, "checkpoint": str(path)})
        for stage in ("base", "epoch1", "epoch2"):
            self.cache(stage)
        return path

    def test_no_cache_means_nothing_to_reuse_and_incomplete_training_is_not_skipped(self):
        workflow.validate_loss_caches()
        self.partition_loader.assert_not_called()
        self.assertFalse(workflow.completed_stage("train", "unused"))

    def test_valid_loss_cache_preserves_fixed_denominators_without_gpu(self):
        self.cache()
        workflow.validate_loss_caches()
        self.partition_loader.assert_called_once_with("validation", self.run)

    def test_corrupt_loss_hash_and_missing_loss_refuse_before_launch(self):
        folder, _ = self.cache()
        loss = folder / "loss.jsonl"
        with loss.open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(AssertionError, "validation loss changed"):
            workflow.launch("train")
        loss.unlink()
        with self.assertRaises(AssertionError):
            workflow.validate_loss_caches()

    def test_correct_hash_cannot_hide_wrong_order_duplicate_or_missing_ids(self):
        folder, original = self.cache()
        for corruption in ("reordered", "duplicate", "missing"):
            with self.subTest(corruption=corruption):
                records = [dict(r) for r in original]
                if corruption == "reordered":
                    records[0], records[1] = records[1], records[0]
                elif corruption == "duplicate":
                    records[0]["id"] = records[1]["id"]
                else:
                    records.pop()
                self.rehash_cache(folder, records)
                with self.assertRaises(AssertionError):
                    workflow.validate_loss_caches()

    def test_same_hash_and_task_counts_cannot_hide_wrong_per_id_task(self):
        folder, records = self.cache()
        records[0]["task"], records[228]["task"] = records[228]["task"], records[0]["task"]
        self.assertEqual(sum(r["task"] == "explain" for r in records), 228)
        self.rehash_cache(folder, records)
        with self.assertRaises(AssertionError):
            workflow.validate_loss_caches()

    def test_completed_training_checks_actual_checkpoint_and_loss_cache(self):
        path = self.checkpoint()
        self.assertTrue(workflow.completed_stage("train", "unused"))
        (path / checkpoints.RANK_FILES[1]).write_bytes(b"corrupted rank-local checkpoint")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            workflow.completed_stage("train", "unused")

    def test_completed_training_refuses_correct_identity_but_wrong_checkpoint_step(self):
        self.checkpoint(step=909)
        with self.assertRaises(AssertionError):
            workflow.completed_stage("train", "unused")

    def test_completed_training_refuses_rehashed_validation_id_corruption(self):
        self.checkpoint()
        folder = self.run / "validation/epoch2"
        records = [json.loads(s) for s in (folder / "loss.jsonl").read_text().splitlines()]
        records[0]["id"] = "foreign-validation-id"
        self.rehash_cache(folder, records)
        with self.assertRaises(AssertionError):
            workflow.completed_stage("train", "unused")


if __name__ == "__main__":
    unittest.main()
