"""Runtime and resumable-checkpoint audits for the authorized two-GPU SFT run."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DistributedSampler
from transformers import TrainerCallback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


class EpochBatchSampler(BatchSampler):
    """Preserve DistributedSampler's epoch seed when Swift rebuilds a skip loader."""

    def __init__(self, sampler, batch_size=1, skip_batches=0):
        super().__init__(sampler, batch_size=batch_size, drop_last=False)
        self.skip_batches = skip_batches

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.skip_batches:
                yield batch

    def __len__(self):
        return max(0, super().__len__() - self.skip_batches)


def padding_provenance(metadata, epochs=2, world_size=2, seed=20261003):
    """Report every deliberate sampler padding replay without duplicating files."""
    count = len(metadata)
    padded_total = math.ceil(count / world_size) * world_size
    output = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(count, generator=generator).tolist()
        extras = permutation[:padded_total - count]
        output.append({"epoch": epoch + 1, "sampler_epoch": epoch, "dataset_rows": count,
                       "distributed_draws": padded_total, "padding_replay_count": len(extras),
                       "replayed_rows": [{"dataset_index": index, "id": metadata[index]["id"],
                                          "task": metadata[index]["task"],
                                          "participant": metadata[index]["participant"]}
                                         for index in extras]})
    return output


def install_distributed_loader(trainer, metadata, seed=20261003):
    """Run-local sampler override; never modify installed Swift or dataset files."""
    from swift.llm import DataLoaderShard
    from swift.utils import seed_worker
    from functools import partial

    assert len(metadata) == len(trainer.train_dataset) == 14543
    assert trainer.args.world_size == 2 and trainer.args.per_device_train_batch_size == 1
    assert not trainer.args.dataloader_drop_last

    def get_loader(self, skip_batches=0):
        args = self.args
        sampler = DistributedSampler(self.train_dataset, num_replicas=args.world_size,
                                     rank=args.process_index, shuffle=True, seed=seed, drop_last=False)
        # HF sets epoch on the ordinary loader. Swift's resume skip shim builds
        # a new loader, so initialize it to the checkpoint's current epoch.
        epoch = int(math.floor(getattr(self.state, "epoch", None) or 0))
        sampler.set_epoch(epoch)
        batch_sampler = EpochBatchSampler(sampler, batch_size=1, skip_batches=skip_batches)
        params = {"batch_sampler": batch_sampler, "collate_fn": self.data_collator,
                  "num_workers": args.dataloader_num_workers,
                  "pin_memory": args.dataloader_pin_memory,
                  "persistent_workers": args.dataloader_persistent_workers,
                  "worker_init_fn": partial(seed_worker, num_workers=args.dataloader_num_workers,
                                             rank=args.process_index)}
        if args.dataloader_num_workers > 0 and args.dataloader_prefetch_factor is not None:
            params["prefetch_factor"] = args.dataloader_prefetch_factor
        return DataLoaderShard(self.train_dataset, device=self.accelerator.device, **params)

    trainer.get_train_dataloader = MethodType(get_loader, trainer)
    loader = trainer.get_train_dataloader()
    effective_batch = trainer.args.per_device_train_batch_size * trainer.args.world_size * trainer.args.gradient_accumulation_steps
    values = trainer.set_initial_training_values(trainer.args, loader, effective_batch)
    epochs, updates, examples, samples, epoch_based, length, max_steps = values
    assert length == 7272 and updates == 909 and max_steps == 1818 and epochs == 2 and epoch_based
    assert effective_batch == 16
    return {"dataset_records": len(metadata), "world_size": 2, "per_device_batch_size": 1,
            "gradient_accumulation_steps": 8, "effective_batch_size": effective_batch,
            "per_rank_batches_per_epoch": length, "optimizer_steps_per_epoch": updates,
            "num_train_epochs": epochs, "expected_optimizer_steps": max_steps,
            "final_accumulation_microbatches": length % 8 or 8,
            "hf_num_examples": examples, "hf_num_train_samples": samples,
            "sampler": "torch.utils.data.DistributedSampler(shuffle=True,drop_last=False)",
            "dataset_load_order_fixed": True, "sampler_seed": seed,
            "override_reason": "Swift3.12.6 default BatchSamplerShard truncates one of14543 rows at world_size2",
            "padding": padding_provenance(metadata, epochs=epochs, world_size=2, seed=seed)}


def checkpoint_files(world_size=2):
    return ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
            "optimizer.pt", "scheduler.pt"] + [f"rng_state_{rank}.pth" for rank in range(world_size)]


def validate_checkpoint(path, config_sha256=None, require_marker=True):
    """Reject partial or incompatible checkpoints without loading model weights."""
    path = Path(path)
    marker = path / "sft_checkpoint_complete.json"
    if require_marker and not marker.is_file():
        raise ValueError(f"Checkpoint has no completion manifest: {path}")
    saved = json.loads(marker.read_text()) if marker.is_file() else None
    state = json.loads((path / "trainer_state.json").read_text())
    step = int(state["global_step"])
    if path.name != f"checkpoint-{step}":
        raise ValueError(f"Checkpoint directory/step mismatch: {path}")
    for name in checkpoint_files():
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Missing or empty resumable checkpoint file: {file}")
    if saved:
        if saved["global_step"] != step or saved["world_size"] != 2:
            raise ValueError(f"Checkpoint completion manifest mismatch: {path}")
        if set(saved["files"]) != set(checkpoint_files()):
            raise ValueError(f"Checkpoint manifest omits resumable state files: {path}")
        if config_sha256 is not None and saved["frozen_config_sha256"] != config_sha256:
            raise ValueError(f"Checkpoint belongs to a different frozen configuration: {path}")
        for name, info in saved["files"].items():
            if (path / name).stat().st_size != info["bytes"] or file_sha256(path / name) != info["sha256"]:
                raise ValueError(f"Checkpoint file hash mismatch: {path / name}")
    return {"path": str(path.resolve()), "global_step": step, "epoch": state.get("epoch"),
            "complete_manifest": saved}


def select_resume_checkpoint(output_dir, config_sha256):
    """Resume the newest valid complete checkpoint; never silently restart progress."""
    output_dir = Path(output_dir)
    candidates = sorted(output_dir.glob("checkpoint-*"),
                        key=lambda path: int(path.name.rsplit("-", 1)[1]) if path.name.rsplit("-", 1)[1].isdigit() else -1,
                        reverse=True)
    rejected = []
    for path in candidates:
        try:
            checkpoint = validate_checkpoint(path, config_sha256)
            return checkpoint, rejected
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
    if candidates:
        raise RuntimeError(f"Existing checkpoints are all invalid; refusing a fresh restart: {rejected}")
    return None, rejected


class TrainingAuditCallback(TrainerCallback):
    """Finite local losses/gradients, retained epoch snapshots and both GPU metrics."""

    def __init__(self, run_dir, trainer, metadata, frozen_config_sha256, schedule, resume=None):
        self.run = Path(run_dir)
        self.trainer = trainer
        self.metadata = metadata
        self.config_sha256 = frozen_config_sha256
        self.schedule = schedule
        self.resume = resume
        self.rank = trainer.args.process_index
        self.metrics_file = self.run / f"train_metrics_rank{self.rank}.jsonl"
        self.microbatch_file = self.run / f"microbatch_loss_rank{self.rank}.jsonl"
        self.start_step = resume["global_step"] if resume else 0
        self.epoch_snapshot_steps = set()
        self.microbatches = 0
        self.pending_losses = []
        self.start_hashes = {}
        self.started = None
        self.session_id = None

    def install_loss_guard(self):
        original = self.trainer.compute_loss

        def compute_loss(model, inputs, *args, **kwargs):
            grid = inputs["image_grid_thw"]
            assert tuple(grid.shape) == (16, 3), tuple(grid.shape)
            assert inputs["input_ids"].shape[-1] <= 8192
            assert (inputs["labels"] != -100).any().item()
            result = original(model, inputs, *args, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                append_jsonl(self.microbatch_file, {"finite": False, "loss": value,
                                                   "global_step": self.trainer.state.global_step,
                                                   "rank": self.rank})
                raise FloatingPointError(f"Nonfinite local loss on rank{self.rank}: {value}")
            self.microbatches += 1
            self.pending_losses.append(value)
            return result

        self.trainer.compute_loss = compute_loss

    def on_train_begin(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        assert state.global_step == self.start_step
        assert args.bf16 and args.world_size == 2 and args.per_device_train_batch_size == 1
        assert args.gradient_accumulation_steps == 8 and state.max_steps == 1818
        assert args.num_train_epochs == 2 and args.max_steps < 0
        assert not args.logging_nan_inf_filter and not args.save_only_model and args.save_total_limit is None
        assert args.save_steps == 250 and args.seed == args.data_seed == 20261003
        assert not args.push_to_hub and str(args.eval_strategy).lower().endswith("no")
        assert str(args.lr_scheduler_type).lower().endswith("cosine")
        assert args.learning_rate == 1e-4 and args.warmup_ratio == .03
        assert str(args.optim).lower().endswith("adamw_torch")
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
        assert args.train_dataloader_shuffle and not args.ignore_data_skip and not args.resume_only_model
        assert optimizer is not None and bool(getattr(model, "is_gradient_checkpointing", False))
        gc = {name: bool(module.gradient_checkpointing) for name, module in model.named_modules()
              if hasattr(module, "gradient_checkpointing")}
        assert any(value for name, value in gc.items() if "language_model" in name)
        assert not any(value for name, value in gc.items() if ".visual" in name)
        gc_kwargs = {name: getattr(getattr(module, "_gradient_checkpointing_func", None), "keywords", None)
                     for name, module in model.named_modules()
                     if "language_model" in name and getattr(module, "gradient_checkpointing", False)}
        assert gc_kwargs and all(value is not None and value.get("use_reentrant") is False
                                 for value in gc_kwargs.values())
        assert model.config.text_config.use_cache is False
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert len(trainable) == 504 and sum(p.numel() for p in trainable.values()) == 43646976
        assert all((".lora_A." in name or ".lora_B." in name) and "language_model" in name
                   and not any(word in name for word in ("visual", "merger", "aligner")) for name in trainable)
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in trainable.values()}
        underlying_optimizer = getattr(optimizer, "optimizer", optimizer)
        assert isinstance(underlying_optimizer, torch.optim.AdamW)
        assert underlying_optimizer.defaults["betas"] == (.9, .95)
        assert underlying_optimizer.defaults["eps"] == 1e-8
        assert lr_scheduler is not None
        scheduler_state = lr_scheduler.state_dict()
        assert scheduler_state["last_epoch"] == self.start_step
        assert args.get_warmup_steps(state.max_steps) == 55
        if self.start_step == 0:
            assert not underlying_optimizer.state
        else:
            assert len(underlying_optimizer.state) == 504
        config = model.peft_config["default"]
        assert config.r == 16 and config.lora_alpha == 32 and config.lora_dropout == 0
        self.start_hashes = {name: tensor_sha256(parameter) for name, parameter in trainable.items()}
        if dist.is_initialized():
            peer = [None] * 2
            dist.all_gather_object(peer, self.start_hashes)
            assert peer[0] == peer[1], "DDP replicas did not start from identical LoRA tensors"
        session = [datetime.now(timezone.utc).isoformat() if self.rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(session, src=0)
        self.session_id = session[0]
        write_json(self.run / f"resolved_training_rank{self.rank}.json", {
            "frozen_config_sha256": self.config_sha256, "rank": self.rank, "world_size": args.world_size,
            "model_class": type(model).__name__, "resolved_trainer_args": vars(args),
            "model_checkpoint": str(model.model_info.model_dir), "schedule": self.schedule,
            "resume": self.resume, "trainable_names": list(trainable), "trainable_parameters": 43646976,
            "lora_rank": config.r, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout,
            "base_dtype": str(model.dtype), "adapter_dtypes": sorted({str(p.dtype) for p in trainable.values()}),
            "checkpointing_modules": gc, "optimizer_only_text_lora": True,
            "actual_text_checkpoint_function_kwargs": gc_kwargs, "text_use_cache": model.config.text_config.use_cache,
            "start_lora_sha256": self.start_hashes,
            "session_id": self.session_id,
            "optimizer_class": type(underlying_optimizer).__name__,
            "optimizer_defaults": underlying_optimizer.defaults,
            "optimizer_group_parameters": [sum(p.numel() for p in group["params"]) for group in optimizer.param_groups],
            "initial_optimizer_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "scheduler_state": scheduler_state, "warmup_optimizer_steps": args.get_warmup_steps(state.max_steps),
        })
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_started = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        norms, missing = [], []
        nonzero = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None, name
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            gradient = parameter.grad.detach()
            norm = float(torch.linalg.vector_norm(gradient.float()).item())
            assert math.isfinite(norm), name
            norms.append(norm)
            nonzero += norm > 0
        assert not missing and len(norms) == 504 and nonzero > 0, missing
        self.last_gradient = {"finite_gradient_tensors": len(norms), "nonzero_gradient_tensors": nonzero,
                              "gradient_l2_after_clipping": math.sqrt(sum(value * value for value in norms)),
                              "lr_before_optimizer": [group["lr"] for group in optimizer.param_groups]}

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.step_started
        assert not self.trainer.accelerator.optimizer_step_was_skipped
        values = self.pending_losses
        assert len(values) == 8 and all(math.isfinite(value) for value in values)
        self.pending_losses = []
        append_jsonl(self.metrics_file, {"rank": self.rank, "session_id": self.session_id,
                     "session_start_step": self.start_step, "global_step": state.global_step,
                     "epoch": state.epoch, "microbatches_this_optimizer_step": len(values),
                     "local_microbatch_losses": values, "mean_local_loss": sum(values) / len(values),
                     **self.last_gradient, "seconds_per_optimizer_step": seconds,
                     "local_examples_per_second": len(values) / seconds,
                     "elapsed_session_seconds": time.perf_counter() - self.started,
                     "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                     "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "cuda_device_index": torch.cuda.current_device(), "gpu_name": torch.cuda.get_device_name()})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            assert math.isfinite(float(logs["loss"]))
            append_jsonl(self.run / f"framework_logs_rank{self.rank}.jsonl",
                         {"global_step": state.global_step, "epoch": state.epoch, "rank": self.rank, **logs})

    def on_epoch_end(self, args, state, control, **kwargs):
        rounded = round(state.epoch or 0)
        if rounded in (1, 2) and math.isclose(state.epoch, rounded, abs_tol=1e-8):
            assert state.global_step == rounded * self.schedule["optimizer_steps_per_epoch"]
            self.epoch_snapshot_steps.add(state.global_step)
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if dist.is_initialized():
            dist.barrier()
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if self.rank == 0:
            validate_checkpoint(checkpoint, require_marker=False)
            files = {name: {"bytes": (checkpoint / name).stat().st_size,
                            "sha256": file_sha256(checkpoint / name)} for name in checkpoint_files()}
            manifest = {"global_step": state.global_step, "epoch": state.epoch, "world_size": 2,
                        "frozen_config_sha256": self.config_sha256, "files": files,
                        "epoch_boundary": state.global_step in self.epoch_snapshot_steps,
                        "fully_resumable": True, "retention": "all checkpoints retained; no rotation"}
            write_json(checkpoint / "sft_checkpoint_complete.json", manifest)
            index = self.run / "checkpoint_index.json"
            previous = json.loads(index.read_text()) if index.is_file() else {"checkpoints": []}
            previous["checkpoints"] = [row for row in previous["checkpoints"] if row["global_step"] != state.global_step]
            previous["checkpoints"].append({"path": str(checkpoint), **manifest})
            previous["checkpoints"].sort(key=lambda row: row["global_step"])
            previous["frozen_config_sha256"] = self.config_sha256
            previous["latest"] = str(checkpoint)
            if manifest["epoch_boundary"]:
                previous[f"epoch{round(state.epoch)}"] = str(checkpoint)
            write_json(index, previous)
        if dist.is_initialized():
            dist.barrier()

    def on_train_end(self, args, state, control, model=None, **kwargs):
        assert state.global_step == self.schedule["expected_optimizer_steps"] and math.isclose(state.epoch, 2, abs_tol=1e-8)
        final = validate_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", self.config_sha256)
        after = {name: tensor_sha256(parameter) for name, parameter in model.named_parameters() if parameter.requires_grad}
        changed = [name for name in after if after[name] != self.start_hashes[name]]
        assert changed or self.start_step == self.schedule["expected_optimizer_steps"]
        index = json.loads((self.run / "checkpoint_index.json").read_text())
        for epoch in (1, 2):
            valid_epoch = validate_checkpoint(index[f"epoch{epoch}"], self.config_sha256)
            assert valid_epoch["global_step"] == epoch * self.schedule["optimizer_steps_per_epoch"]
            assert math.isclose(valid_epoch["epoch"], epoch, abs_tol=1e-8)
        write_json(self.run / f"training_complete_rank{self.rank}.json", {
            "passed": True, "frozen_config_sha256": self.config_sha256,
            "global_step": state.global_step, "epoch": state.epoch,
            "session_start_step": self.start_step, "optimizer_steps_this_session": state.global_step - self.start_step,
            "finite_local_loss_checks": self.microbatches, "lora_tensors_changed_this_session": len(changed),
            "after_lora_sha256": after, "final_checkpoint": final["path"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "session_seconds": time.perf_counter() - self.started, "rank": self.rank,
            "session_id": self.session_id,
        })


def run_cpu_self_tests(output=None):
    """Exercise actual installed loader/scheduling and interrupted-save handling."""
    import tempfile
    from types import SimpleNamespace
    from transformers import Trainer

    if torch.cuda.is_initialized():
        raise RuntimeError("Self-tests must run without initialized CUDA")
    metadata = [{"id": f"row-{i}", "task": "explain" if i % 10 == 0 else "track",
                 "participant": "P01"} for i in range(14543)]
    args = SimpleNamespace(world_size=2, process_index=0, per_device_train_batch_size=1,
                           gradient_accumulation_steps=8, dataloader_drop_last=False,
                           dataloader_num_workers=0, dataloader_pin_memory=False,
                           dataloader_persistent_workers=False, dataloader_prefetch_factor=None,
                           num_train_epochs=2, max_steps=-1)
    fake = SimpleNamespace(args=args, state=SimpleNamespace(epoch=0),
                           train_dataset=list(range(14543)), data_collator=lambda rows: rows,
                           accelerator=SimpleNamespace(device=torch.device("cpu")))
    fake.num_examples = lambda loader: len(loader.dataset)
    fake.set_initial_training_values = MethodType(Trainer.set_initial_training_values, fake)
    schedule = install_distributed_loader(fake, metadata)
    assert schedule["expected_optimizer_steps"] == 1818
    provenance = padding_provenance(metadata)
    for epoch in (0, 1):
        ranks = []
        for rank in (0, 1):
            sampler = DistributedSampler(fake.train_dataset, num_replicas=2, rank=rank,
                                         shuffle=True, seed=20261003, drop_last=False)
            sampler.set_epoch(epoch)
            ranks.append(list(sampler))
        flattened = ranks[0] + ranks[1]
        counts = Counter(flattened)
        assert len(ranks[0]) == len(ranks[1]) == 7272
        assert set(flattened) == set(range(14543)) and sum(counts.values()) == 14544
        assert sorted(counts.values()).count(2) == 1 and max(counts.values()) == 2
        replayed = next(index for index, count in counts.items() if count == 2)
        assert provenance[epoch]["replayed_rows"][0]["dataset_index"] == replayed
        if epoch == 1:
            fake.state.epoch = 1.25
            resumed = fake.get_train_dataloader(skip_batches=2000)
            assert len(resumed) == 5272
            assert [batch[0] for batch in resumed.batch_sampler] == ranks[0][2000:]
            resumed.set_epoch(0)
            expected = DistributedSampler(fake.train_dataset, num_replicas=2, rank=0,
                                          shuffle=True, seed=20261003, drop_last=False)
            assert [batch[0] for batch in resumed.batch_sampler] == list(expected)[2000:]

    with tempfile.TemporaryDirectory(prefix="hdepic-train-audit-") as temporary:
        root = Path(temporary)
        fake.args.output_dir = str(root / "checkpoints")
        callback = TrainingAuditCallback(root, fake, metadata, "test-config-hash", schedule)
        control = SimpleNamespace(should_save=False)
        for epoch, step in ((1.0, 909), (2.0, 1818)):
            state = SimpleNamespace(epoch=epoch, global_step=step)
            callback.on_epoch_end(fake.args, state, control)
            assert control.should_save
            checkpoint = Path(fake.args.output_dir) / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            for name in checkpoint_files():
                (checkpoint / name).write_bytes(b"fake-state-" + name.encode())
            write_json(checkpoint / "trainer_state.json", {"global_step": step, "epoch": epoch})
            callback.on_save(fake.args, state, control)
            assert validate_checkpoint(checkpoint, "test-config-hash")["global_step"] == step
        index = json.loads((root / "checkpoint_index.json").read_text())
        assert index["epoch1"].endswith("checkpoint-909") and index["epoch2"].endswith("checkpoint-1818")
        latest, rejected = select_resume_checkpoint(fake.args.output_dir, "test-config-hash")
        assert latest["global_step"] == 1818 and not rejected
        # A partial later save must fall back explicitly to the retained valid save.
        partial = Path(fake.args.output_dir) / "checkpoint-1900"
        partial.mkdir()
        latest, rejected = select_resume_checkpoint(fake.args.output_dir, "test-config-hash")
        assert latest["global_step"] == 1818 and len(rejected) == 1
        # Corruption of a complete checkpoint is detected rather than loaded.
        (Path(index["epoch2"]) / "optimizer.pt").write_bytes(b"corrupt")
        latest, rejected = select_resume_checkpoint(fake.args.output_dir, "test-config-hash")
        assert latest["global_step"] == 909 and len(rejected) == 2
        (Path(index["epoch1"]) / "scheduler.pt").write_bytes(b"corrupt")
        try:
            select_resume_checkpoint(fake.args.output_dir, "test-config-hash")
        except RuntimeError:
            pass
        else:
            raise AssertionError("All-invalid checkpoints silently restarted")
        # A saved completion marker cannot change experiment identity.
        try:
            validate_checkpoint(index["epoch1"], "other-config-hash")
        except ValueError:
            pass
        else:
            raise AssertionError("Checkpoint/config identity mismatch was accepted")
    result = {"passed": True, "cuda_initialized": torch.cuda.is_initialized(),
              "actual_installed_loader": "swift.llm.DataLoaderShard",
              "actual_hf_schedule": "transformers.Trainer.set_initial_training_values",
              "expected_optimizer_steps": 1818, "per_rank_batches_per_epoch": 7272,
              "optimizer_steps_per_epoch": 909,
              "checks": ["all14543 rows covered with one explicit padding replay each epoch",
                         "padding replay provenance matches actual DistributedSampler",
                         "resume skip preserves epoch1 shuffle and skipped batch count",
                         "loader set_epoch forwards through batch sampler",
                         "epoch-boundary callback retains epoch1 andepoch2 snapshots",
                         "newer incomplete checkpoint is rejected with explicit fallback",
                         "corrupted optimizer/scheduler hashes are rejected",
                         "all-invalid checkpoints refuse fresh restart",
                         "frozen configuration identity mismatch is rejected"]}
    assert not result["cuda_initialized"]
    if output:
        write_json(output, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    print(json.dumps(run_cpu_self_tests(options.output), indent=2))
