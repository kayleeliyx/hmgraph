"""CPU-only allocation, stage-resume and frozen-identity orchestration guards."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from code.hdepic_experiment import prepare, workflow
from code.hdepic_experiment.common import atomic_json, atomic_jsonl, sha256, stable_hash


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        atomic_json(self.run / "frozen_config.json", {"identity": "frozen-config"})
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.temp.cleanup()

    def test_completed_baseline_verifies_cache_without_gpu_allocation(self):
        with mock.patch.object(workflow, "evaluation_complete", return_value=True), mock.patch.object(workflow, "child") as child, mock.patch.object(workflow, "idle_evaluation_device") as idle:
            workflow.evaluate(self.run, "base")
            idle.assert_not_called()
            self.assertEqual(child.call_args.args[1], "base_verify_cache")
            self.assertEqual(child.call_args.args[3], [0])
            self.assertIn("base", child.call_args.args[2])

    def test_new_baseline_waits_for_allocated_pair_and_verifies_after_child(self):
        with mock.patch.object(workflow, "evaluation_complete", side_effect=[False, True]) as complete, mock.patch.object(workflow, "child") as child, mock.patch.object(workflow, "idle_evaluation_device", return_value=1) as idle:
            workflow.evaluate(self.run, "base")
            idle.assert_called_once_with(self.run, "base", require_pair=True)
            self.assertEqual(child.call_args.args[3], [1])
            self.assertEqual(complete.call_count, 2)

    def test_epoch_evaluation_is_blocked_before_both_epochs_complete(self):
        with mock.patch.object(workflow, "training_complete", return_value=False), mock.patch.object(workflow, "child") as child, mock.patch.object(workflow, "idle_evaluation_device") as idle:
            with self.assertRaises(AssertionError):
                workflow.evaluate(self.run, "epoch1")
            child.assert_not_called(); idle.assert_not_called()

    def test_completed_epoch_verifies_the_indexed_checkpoint_cache(self):
        with mock.patch.object(workflow, "training_complete", return_value=True), mock.patch.object(workflow, "evaluation_complete", return_value=True), mock.patch.object(workflow, "read_json", return_value={"epoch1": "/expected/checkpoint-909"}), mock.patch.object(workflow, "child") as child, mock.patch.object(workflow, "idle_evaluation_device") as idle:
            workflow.evaluate(self.run, "epoch1")
            argv = child.call_args.args[2]
            self.assertEqual(argv[argv.index("--checkpoint") + 1], "/expected/checkpoint-909")
            self.assertEqual(child.call_args.args[1], "epoch1_verify_cache")
            idle.assert_not_called()

    def test_training_requires_base_then_skips_already_complete_without_gpu(self):
        with mock.patch.object(workflow, "evaluation_complete", return_value=False), mock.patch.object(workflow, "training_complete") as completed, mock.patch.object(workflow, "idle_pair") as idle, mock.patch.object(workflow, "child") as child:
            with self.assertRaises(AssertionError):
                workflow.train(self.run)
            completed.assert_not_called(); idle.assert_not_called(); child.assert_not_called()
        with mock.patch.object(workflow, "evaluation_complete", return_value=True), mock.patch.object(workflow, "training_complete", return_value=True), mock.patch.object(workflow, "idle_pair") as idle, mock.patch.object(workflow, "child") as child:
            workflow.train(self.run)
            idle.assert_not_called(); child.assert_not_called()

    def test_occupied_allocated_devices_stop_before_child(self):
        with mock.patch.object(workflow, "evaluation_complete", return_value=True), mock.patch.object(workflow, "training_complete", return_value=False), mock.patch.object(workflow, "idle_pair", side_effect=RuntimeError("occupied")), mock.patch.object(workflow, "child") as child:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                workflow.train(self.run)
            child.assert_not_called()

    def test_installed_torchrun_parser_preserves_training_module_arguments(self):
        import torch
        from torch.distributed.run import parse_args
        initialized_before = torch.cuda.is_initialized()
        with mock.patch.object(workflow, "evaluation_complete", return_value=True), mock.patch.object(workflow, "training_complete", side_effect=[False, True]), mock.patch.object(workflow, "idle_pair"), mock.patch.object(workflow, "child", return_value={"mock_execution": True}) as child, mock.patch.object(workflow, "read_json", return_value={"rank": 0}):
            workflow.train(self.run)
        argv = child.call_args.args[2]
        self.assertEqual(argv[:3], [str(workflow.PYTHON), "-m", "torch.distributed.run"])
        parsed = parse_args(argv[3:])
        self.assertTrue(parsed.module)
        self.assertEqual(parsed.nproc_per_node, "2")
        self.assertEqual(parsed.training_script, "code.hdepic_experiment.train")
        self.assertEqual(parsed.training_script_args, ["--run", str(self.run), "--config", str(self.run / "frozen_config.json")])
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)

    def test_installed_torchrun_reproduces_unseparated_run_argument_ambiguity(self):
        from torch.distributed.run import parse_args
        args = ["--standalone", "--nproc_per_node", "2", "-m", "code.hdepic_experiment.train", "--run", str(self.run), "--config", str(self.run / "frozen_config.json")]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as failure:
            parse_args(args)
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("ambiguous option: --run", stderr.getvalue())

    def test_evaluation_falls_back_only_to_the_other_authorized_gpu(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(workflow, "require_idle_gpus", side_effect=[RuntimeError("GPU0 occupied"), {"idle": True}]) as idle:
            device = workflow.idle_evaluation_device(self.run, "epoch1")
            self.assertEqual(device, 1)
            self.assertEqual([call.args[0] for call in idle.call_args_list], [(0,), (1,)])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], workflow.GPU_UUIDS[1])

    def test_subprocess_error_preserves_failure_manifest_and_never_runs_command(self):
        with mock.patch.object(workflow.subprocess, "run", return_value=subprocess.CompletedProcess(["mock"], 7)) as child:
            with self.assertRaisesRegex(RuntimeError, "exit7"):
                workflow.child(self.run, "cpu_stage", ["this-is-never-executed"])
            self.assertEqual(child.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            manifests = list((self.run / "logs").glob("*.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())["exit_code"], 7)
            self.assertTrue(list((self.run / "logs").glob("*.log")))

    def test_exclusive_workflow_lock_blocks_stage_work(self):
        with mock.patch.object(workflow, "RUN", self.run), mock.patch.object(workflow, "configure_environment"), mock.patch.object(workflow.fcntl, "flock", side_effect=BlockingIOError("already running")), mock.patch.object(workflow, "prepare") as preparation, mock.patch.object(workflow, "evaluate") as evaluate, mock.patch("sys.argv", ["workflow", "all", "--run", str(self.run)]):
            with self.assertRaises(BlockingIOError):
                workflow.main()
            preparation.assert_not_called(); evaluate.assert_not_called()

    def test_all_stages_have_fixed_sequential_order(self):
        calls = []
        with mock.patch.object(workflow, "RUN", self.run), mock.patch.object(workflow, "configure_environment"), mock.patch.object(workflow, "prepare", side_effect=lambda _: calls.append("prepare")), mock.patch.object(workflow, "evaluate", side_effect=lambda _, stage, retry: calls.append(stage)), mock.patch.object(workflow, "train", side_effect=lambda _: calls.append("train")), mock.patch.object(workflow, "report", side_effect=lambda _: calls.append("report")), mock.patch("sys.argv", ["workflow", "all", "--run", str(self.run)]):
            workflow.main()
        self.assertEqual(calls, ["prepare", "base", "train", "epoch1", "epoch2", "report"])

    def completed_base_fixture(self, adapter=None):
        identities = [f"validation-{i}" for i in range(2209)]
        sources = {}
        for kind in ("prompts", "export", "scoring"):
            path = self.run / (kind + ".jsonl")
            atomic_jsonl(path, [{"row": i, "kind": kind} for i in range(2209)])
            sources[kind] = str(path)
        config = {"data": {"validation": sources}, "model": {"path": "/model", "revision": "revision"}}
        atomic_json(self.run / "frozen_config.json", config)
        atomic_json(self.run / "frozen_evaluation.json", {"ordered_validation_ids": identities})
        config_sha = sha256(self.run / "frozen_config.json")
        evaluation_sha = sha256(self.run / "frozen_evaluation.json")
        model = {"stage": "base", "revision": "revision", "original_path": "/model", "adapter": adapter}
        model["identity_sha256"] = stable_hash(model)
        identity = {"stage": "base", "frozen_config_sha256": config_sha, "frozen_evaluation_sha256": evaluation_sha, "data": sources, "model": model, "code_sha256": {}}
        run_key = stable_hash(identity)
        folder = self.run / "predictions/base"
        atomic_json(folder / "metadata.json", {"identity": identity, "run_key": run_key})
        from code.hdepic_experiment.eval import source_line_hashes
        source_hashes = {k: source_line_hashes(v) for k, v in sources.items()}
        rows = []
        for index, ident in enumerate(identities):
            key = stable_hash({"id": ident, "row_index": index, "prompt_sha256": source_hashes["prompts"][index], "export_sha256": source_hashes["export"][index], "scoring_sha256": source_hashes["scoring"][index], "run_key": run_key})
            rows.append({"id": ident, "row_index": index, "partition": "validation", "stage": "base", "status": "completed", "run_key": run_key, "record_key": key, "prompt_sha256": source_hashes["prompts"][index], "model_identity_sha256": model["identity_sha256"], "frozen_config_sha256": config_sha, "frozen_evaluation_sha256": evaluation_sha})
        for kind in ("generation", "loss"):
            atomic_jsonl(folder / (kind + ".jsonl"), rows)
        marker = {"passed": True, "stage": "base", "partition": "validation", "records": 2209, "generation_records": 2209, "loss_records": 2209, "frozen_config_sha256": config_sha, "frozen_evaluation_sha256": evaluation_sha, "run_key": run_key, "model_identity_sha256": model["identity_sha256"], "generation_sha256": sha256(folder / "generation.jsonl"), "loss_sha256": sha256(folder / "loss.jsonl")}
        atomic_json(folder / "completed.json", marker)
        return config, folder, rows, marker

    def test_completion_accepts_only_matching_stage_model_and_ordered_ids(self):
        config, folder, rows, marker = self.completed_base_fixture()
        with mock.patch.object(workflow, "load_config", return_value=config):
            self.assertTrue(workflow.evaluation_complete(self.run, "base"))
            atomic_json(folder / "completed.json", {**marker, "stage": "epoch1"})
            with self.assertRaises(AssertionError):
                workflow.evaluation_complete(self.run, "base")
            atomic_json(folder / "completed.json", marker)
            rows[0], rows[1] = rows[1], rows[0]
            atomic_jsonl(folder / "generation.jsonl", rows)
            atomic_json(folder / "completed.json", {**marker, "generation_sha256": sha256(folder / "generation.jsonl")})
            with self.assertRaises(AssertionError):
                workflow.evaluation_complete(self.run, "base")

    def test_base_completion_rejects_adapter_and_cross_run_row(self):
        config, folder, rows, marker = self.completed_base_fixture(adapter={"path": "/foreign/adapter"})
        with mock.patch.object(workflow, "load_config", return_value=config), self.assertRaises(AssertionError):
            workflow.evaluation_complete(self.run, "base")
        config, folder, rows, marker = self.completed_base_fixture()
        rows[0]["run_key"] = "foreign-run"
        atomic_jsonl(folder / "generation.jsonl", rows)
        atomic_json(folder / "completed.json", {**marker, "generation_sha256": sha256(folder / "generation.jsonl")})
        with mock.patch.object(workflow, "load_config", return_value=config), self.assertRaises(AssertionError):
            workflow.evaluation_complete(self.run, "base")


class PrepareIdentityTests(unittest.TestCase):
    def test_original_checkpoint_and_source_image_are_rechecked_on_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); run, model, dataset = root / "run", root / "model", root / "data"
            for path in (run, model, dataset):
                path.mkdir()
            weights = model / "weights.bin"; weights.write_bytes(b"original")
            image = root / "source.jpg"; image.write_bytes(b"original")
            image_metadata = {"path": str(image), "bytes": image.stat().st_size, "sha256": sha256(image)}
            atomic_json(model / "generation_config.json", {"eos_token_id": [151645, 151643], "pad_token_id": 151643})
            identity_path = root / "identity.json"
            atomic_json(identity_path, {"snapshot_path": str(model), "revision": "revision", "files": [{"filename": weights.name, "bytes": weights.stat().st_size, "sha256": sha256(weights)}]})
            partitions = {}
            data_info = {}
            for part, count in (("initial_sft", 1), ("validation", 50)):
                records = [{"id": f"{part}-{i}", "task": "explain"} for i in range(count)]
                text = dataset / "text" / (part + ".jsonl"); atomic_jsonl(text, records)
                rows = [{"messages": [], "images": []} for _ in records]
                scores = [{"id": r["id"], "frames": [image_metadata]} for r in records]
                prompts = [{"messages": [], "images": []} for _ in records]
                partitions[part] = (rows, scores, prompts)
                data_info[part] = {"count": count, "task_counts": {"explain": count}, "prompts_sha256": "prompt-hash", "scoring_sha256": "scoring-hash"}
            config = {"paths": {"dataset": str(dataset), "python": "mock-python"}, "data": data_info, "training": {}, "evaluation": {}, "model": {"identity_source": str(identity_path), "path": str(model), "revision": "revision"}}
            atomic_json(run / "frozen_config.json", config)
            versions = {"ms-swift": "3.12.6", "transformers": "4.57.1", "torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "peft": "0.17.1", "accelerate": "1.10.1", "qwen-vl-utils": "0.0.14"}
            distributions = [SimpleNamespace(metadata={"Name": k}, version=v) for k, v in versions.items()]
            with mock.patch.object(prepare, "MODEL", model), mock.patch.object(prepare, "load_config", side_effect=lambda path: json.loads(Path(path).read_text())), mock.patch.object(prepare, "load_records", side_effect=lambda _, part: deepcopy(partitions[part])), mock.patch.object(prepare.importlib.metadata, "distributions", return_value=distributions), mock.patch.object(prepare.subprocess, "check_output", return_value="mock-package-freeze\n"), contextlib.redirect_stdout(io.StringIO()):
                prepare.freeze(run, freeze_code=False)
                self.assertTrue((run / "original_model_identity.json").exists())
                weights.write_bytes(b"modified")  # same bytes count; SHA256 must reject it
                with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                    prepare.freeze(run, freeze_code=False)
                weights.write_bytes(b"original")
                image.write_bytes(b"modified")
                with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                    prepare.freeze(run, freeze_code=False)


if __name__ == "__main__":
    unittest.main()
