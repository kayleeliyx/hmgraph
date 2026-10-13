"""CPU-only checkpoint tests; two-rank Gloo exercises genuine DTensor shards.

This does not substitute for the real CUDA FSDP2 integration preflight.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from code.hdepic_sft_qwen38 import checkpoints as cp
from code.hdepic_sft_qwen38.common import canonical, read_json, write_json


def _identity():
    return {"config_sha256": "c" * 64, "model_sha256": "m" * 64, "dataset_sha256": "d" * 64}


def _state(step):
    return {"step": step, "epoch": 0, "microbatch_consumed": step * 8,
            "sampler_state": {"epoch": 0, "seed": 20261003, "shuffle": True}}


def _fixture(path, step, identity=None):
    path.mkdir()
    identity = identity or _identity()
    for name in cp.RANK_FILES + cp.ADAPTER_FILES:
        (path / name).write_bytes((name + str(step)).encode())
    write_json(path / "completed.json", {"format": cp.FORMAT, "passed": True,
        "world_size": 2, "identity": identity,
        "identity_sha256": hashlib.sha256(canonical(identity)).hexdigest(),
        "state": _state(step), "trainable_schema": {"x.lora_A.default.weight": {"shape": [2, 4]}},
        "files": {name: cp._file(path / name) for name in cp.RANK_FILES + cp.ADAPTER_FILES}})


def _worker(rank, directory):
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    root = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{root/'rendezvous'}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("shard",))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        def forward(self, x):
            return self.linear(x)

    def model(sharded=True):
        torch.manual_seed(42)
        value = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["linear"], bias="none"))
        if sharded:
            for name, parameter in list(value.named_parameters()):
                if parameter.requires_grad:
                    prefix, leaf = name.rsplit(".", 1)
                    shard = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
                    setattr(value.get_submodule(prefix), leaf, torch.nn.Parameter(shard))
            # Any accidental whole-model serialization fails loudly.
            def forbidden(*args, **kwargs):
                raise AssertionError("Frozen model.state_dict() must not be called")
            value.state_dict = forbidden
        return value

    def setup(value, lr=0.01, tmax=10):
        opt = torch.optim.AdamW([p for p in value.parameters() if p.requires_grad], lr=lr,
                               betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax)

    def update(value, opt, scheduler, number):
        for index, (name, parameter) in enumerate(cp._trainable(value).items()):
            local = torch.full_like(parameter.to_local(), 0.01 * (number + index + rank))
            parameter.grad = DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
                                                 shape=parameter.shape, stride=parameter.stride())
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)

    def draws():
        return random.random(), float(np.random.rand()), torch.rand(3)

    def require_raises(fn, phrase):
        try:
            fn()
        except (ValueError, RuntimeError) as exc:
            assert phrase in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected rejection: {phrase}")

    try:
        first = model()
        frozen_name, frozen = next((n, p) for n, p in first.named_parameters() if not p.requires_grad)
        original_base = frozen.detach().clone()
        optimizer, scheduler = setup(first)
        update(first, optimizer, scheduler, 1)
        random.seed(900 + rank); np.random.seed(900 + rank); torch.manual_seed(900 + rank)
        checkpoint1 = root / "checkpoint-000001"
        marker = cp.save_checkpoint(first, optimizer, scheduler, checkpoint1, _state(1), _identity())
        assert marker["passed"] and cp.choose_resume(root, _identity()) == checkpoint1
        original_files = {p.name: cp.sha(p) for p in checkpoint1.iterdir()}
        # Re-saving the exact same state is immutable and consumes no RNG.
        cp.save_checkpoint(first, optimizer, scheduler, checkpoint1, _state(1), _identity())
        assert original_files == {p.name: cp.sha(p) for p in checkpoint1.iterdir()}
        exported_full = {n: p.detach().full_tensor().cpu().clone() for n, p in cp._trainable(first).items()}
        expected_rng = draws()
        update(first, optimizer, scheduler, 2)
        expected_params = {n: p.to_local().detach().clone() for n, p in cp._trainable(first).items()}
        expected_opt = {n: cp._pack(optimizer.state[p]) for n, p in cp._trainable(first).items()}
        expected_lr = optimizer.param_groups[0]["lr"]

        resumed = model()
        opt2, sched2 = setup(resumed, lr=0.9, tmax=55)
        restored = cp.load_checkpoint(resumed, opt2, sched2, checkpoint1, _identity())
        assert restored == _state(1)
        assert cp._same(draws(), expected_rng), "Rank Python/NumPy/Torch RNG did not resume exactly"
        update(resumed, opt2, sched2, 2)
        for name, parameter in cp._trainable(resumed).items():
            assert torch.equal(parameter.to_local(), expected_params[name]), name
            assert cp._same(cp._pack(opt2.state[parameter]), expected_opt[name]), name
            assert isinstance(opt2.state[parameter]["exp_avg"], DTensor)
            assert opt2.state[parameter]["step"].device.type == "cpu"
        assert opt2.param_groups[0]["lr"] == expected_lr
        assert sched2.state_dict() == scheduler.state_dict()
        assert torch.equal(dict(resumed.named_parameters())[frozen_name], original_base)
        require_raises(lambda: cp.load_checkpoint(resumed, opt2, sched2, checkpoint1, {"bad": "identity"}), "identity")
        resumed.peft_config["default"].lora_alpha = 999
        require_raises(lambda: cp.load_checkpoint(resumed, opt2, sched2, checkpoint1, _identity()), "configuration changed")
        resumed.peft_config["default"].lora_alpha = 4
        require_raises(lambda: cp.save_checkpoint(resumed, opt2, sched2, checkpoint1, _state(1), _identity()), "differs from current")
        assert original_files == {p.name: cp.sha(p) for p in checkpoint1.iterdir()}

        # Standard PEFT fresh reload needs only the original frozen base and adapter.
        if rank == 0:
            torch.manual_seed(42)
            fresh = PeftModel.from_pretrained(Tiny(), checkpoint1, is_trainable=False, local_files_only=True)
            saved = get_peft_model_state_dict(fresh, save_embedding_layers=False)
            expected = get_peft_model_state_dict(first, state_dict=exported_full, save_embedding_layers=False)
            assert saved.keys() == expected.keys()
            assert all(torch.equal(saved[n], expected[n]) for n in expected)
            assert torch.isfinite(fresh(torch.ones((1, 4), dtype=torch.bfloat16))).all()
            assert all("base_layer" not in n for n in saved)
            assert sum(p.stat().st_size for p in checkpoint1.iterdir()) < 100_000
        dist.barrier()

        checkpoint2 = root / "checkpoint-000002"
        cp.save_checkpoint(resumed, opt2, sched2, checkpoint2, _state(2), _identity())
        assert cp.choose_resume(root, _identity()) == checkpoint2
        before_mismatch = {n: p.to_local().detach().clone() for n, p in cp._trainable(resumed).items()}
        require_raises(lambda: cp.load_checkpoint(resumed, opt2, sched2,
                        checkpoint1 if rank == 0 else checkpoint2, _identity()), "disagree")
        assert all(torch.equal(p.to_local(), before_mismatch[n]) for n, p in cp._trainable(resumed).items())
        mismatch = root / f"checkpoint-path-rank{rank}"
        require_raises(lambda: cp.save_checkpoint(resumed, opt2, sched2, mismatch,
                        _state(2), _identity()), "disagree")
        assert not mismatch.exists() and not mismatch.with_name(mismatch.name + ".incomplete").exists()

        # Saving before clearing accumulated gradients is explicitly rejected.
        parameter = next(iter(cp._trainable(resumed).values()))
        parameter.grad = torch.ones_like(parameter)
        require_raises(lambda: cp.save_checkpoint(resumed, opt2, sched2, root / "dirty-grads",
                        _state(2), _identity()), "zero_grad")
        opt2.zero_grad(set_to_none=True)
        frozen = dict(resumed.named_parameters())[frozen_name]
        frozen.requires_grad_(True)
        require_raises(lambda: cp.save_checkpoint(resumed, opt2, sched2, root / "unfrozen-base",
                        _state(2), _identity()), "Only LoRA")
        frozen.requires_grad_(False)
        if rank == 0:
            with (checkpoint2 / cp.RANK_FILES[1]).open("ab") as handle:
                handle.write(b"corrupt")
        dist.barrier()
        assert cp.choose_resume(root, _identity()) == checkpoint1
        require_raises(lambda: cp.load_checkpoint(resumed, opt2, sched2, checkpoint2, _identity()), "changed")

        # One rank's write failure must leave no completed checkpoint on either rank.
        failing = root / "checkpoint-000003"
        original_save = torch.save
        def injected(*args, **kwargs):
            if rank == 1:
                raise OSError("injected rank1 disk failure")
            return original_save(*args, **kwargs)
        with mock.patch.object(torch, "save", side_effect=injected):
            require_raises(lambda: cp.save_checkpoint(resumed, opt2, sched2, failing, _state(3), _identity()), "rank1 disk failure")
        assert not (failing / "completed.json").exists()
        assert not (failing.with_name(failing.name + ".incomplete") / "completed.json").exists()
        assert cp.choose_resume(root, _identity()) == checkpoint1
        write_json(root / f"rank{rank}_passed.json", {"passed": True, "rank": rank,
                   "exact_next_update": True, "rng": True, "base_untouched": True})
    finally:
        dist.destroy_process_group()


class CheckpointInventoryTests(unittest.TestCase):
    def test_missing_incomplete_wrong_identity_and_corrupt_newer_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(cp.choose_resume(root, _identity()))
            _fixture(root / "checkpoint-1", 1)
            _fixture(root / "checkpoint-2", 2)
            _fixture(root / "checkpoint-99.incomplete", 99)
            _fixture(root / "checkpoint-other", 100, {"other": "identity"})
            self.assertEqual(cp.choose_resume(root, _identity()), root / "checkpoint-2")
            (root / "checkpoint-2" / cp.RANK_FILES[1]).unlink()
            self.assertEqual(cp.choose_resume(root, _identity()), root / "checkpoint-1")
            (root / "checkpoint-1" / cp.ADAPTER_FILES[0]).write_bytes(b"changed")
            self.assertIsNone(cp.choose_resume(root, _identity()))

    def test_unexpected_base_file_inventory_and_marker_digests_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-1"
            _fixture(path, 1)
            (path / "model.safetensors").write_bytes(b"base")
            with self.assertRaisesRegex(ValueError, "Unexpected checkpoint file"):
                cp.validate_checkpoint(path, _identity())
            (path / "model.safetensors").unlink()
            marker = read_json(path / "completed.json")
            marker["identity_sha256"] = "wrong"
            write_json(path / "completed.json", marker)
            with self.assertRaisesRegex(ValueError, "identity"):
                cp.validate_checkpoint(path, _identity())

    def test_requires_complete_progress_and_finite_exact_identity(self):
        for state in ({"step": 1}, dict(_state(1), step=True), dict(_state(1), microbatch_consumed=-1)):
            with self.assertRaises(ValueError):
                cp._progress(state)
        with self.assertRaises(ValueError):
            cp._identity({"model": float("nan")})
        self.assertEqual(cp._progress(_state(1)), _state(1))


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("torch", "peft", "numpy", "safetensors")), "isolated Torch/PEFT environment required")
class DistributedCheckpointTests(unittest.TestCase):
    def test_two_rank_dtensor_exact_resume_and_fresh_peft_reload(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory,), nprocs=2, join=True)
            for rank in range(2):
                self.assertTrue(read_json(Path(directory) / f"rank{rank}_passed.json")["passed"])


if __name__ == "__main__":
    unittest.main()
