"""Frozen, resumable two-GPU initial-SFT run, gated by preflight/base validation.

Invoke with torchrun using only the two allocated GPU UUIDs. The orchestrator
checks that both GPUs are idle before torchrun; per-rank occupancy checks would
race with the other replica loading. This entry never launches evaluation,
second-stage training, inference, or any remote service.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path

from .common import (CONFIG_PATH, GPU_UUIDS, RUN, configure_environment,
                     load_config, load_records, read_json, sha256, stable_hash)


def build_swift_argv(config, output, resume=None):
    """Keep the actual invocation explicit instead of accepting framework defaults."""
    training, processor = config["training"], config["processor"]
    values = {
        "model": config["model"]["path"], "model_type": "qwen3_vl", "use_hf": True,
        "train_type": training["train_type"], "tuner_backend": training["tuner_backend"],
        "target_modules": training["target_modules"],
        "freeze_vit": training["freeze_vit"], "freeze_aligner": training["freeze_aligner"],
        "freeze_llm": training["freeze_llm"], "lora_rank": training["lora_rank"],
        "lora_alpha": training["lora_alpha"], "lora_dropout": training["lora_dropout"],
        "torch_dtype": training["torch_dtype"], "bf16": training["bf16"], "fp16": training["fp16"],
        "attn_impl": processor["attention_implementation"],
        "dataset": config["data"]["initial_sft"]["export"], "split_dataset_ratio": 0,
        "dataset_shuffle": training["dataset_shuffle"],
        "train_dataloader_shuffle": training["train_dataloader_shuffle"],
        "dataset_num_proc": training["dataset_num_proc"],
        "dataloader_num_workers": training["dataloader_num_workers"],
        "dataloader_persistent_workers": training["dataloader_persistent_workers"],
        "dataloader_prefetch_factor": training["dataloader_prefetch_factor"],
        "dataloader_drop_last": False, "lazy_tokenize": True, "strict": processor["strict"],
        "max_length": processor["max_length"], "max_pixels": processor["max_pixels"],
        "truncation_strategy": processor["truncation_strategy"], "loss_scale": processor["loss_scale"],
        "padding_free": training["padding_free"], "packing": training["packing"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": training["gradient_checkpointing_kwargs"],
        "vit_gradient_checkpointing": training["vit_gradient_checkpointing"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "max_steps": training["max_steps"], "num_train_epochs": training["num_train_epochs"],
        "learning_rate": training["learning_rate"], "lr_scheduler_type": training["lr_scheduler_type"],
        "warmup_ratio": training["warmup_ratio"], "optim": training["optim"],
        "weight_decay": training["weight_decay"], "adam_beta1": training["adam_beta1"],
        "adam_beta2": training["adam_beta2"], "adam_epsilon": training["adam_epsilon"],
        "max_grad_norm": training["max_grad_norm"], "logging_steps": training["logging_steps"],
        "logging_nan_inf_filter": training["logging_nan_inf_filter"],
        "eval_strategy": "no", "save_strategy": "steps", "save_steps": training["save_steps"],
        "save_only_model": training["save_only_model"], "save_safetensors": True,
        "output_dir": str(output), "add_version": False, "report_to": "none",
        "seed": training["seed"], "data_seed": training["data_seed"],
        "push_to_hub": False, "ddp_find_unused_parameters": training["ddp_find_unused_parameters"],
        "load_args": False, "resume_only_model": False, "ignore_data_skip": False,
    }
    # Omitting a None save_total_limit preserves every 250-step and epoch save.
    assert training["save_total_limit"] is None
    if resume:
        values["resume_from_checkpoint"] = resume["path"]
    argv = []
    for key, value in values.items():
        argv.append("--" + key)
        if isinstance(value, bool):
            argv.append(str(value).lower())
        elif isinstance(value, list):
            argv.extend(map(str, value))
        elif isinstance(value, dict):
            argv.append(json.dumps(value, separators=(",", ":")))
        else:
            argv.append(str(value))
    return argv


def maximum_observed_step(training):
    """Report uncommitted progress explicitly when resuming a saved step."""
    maximum = 0
    for path in Path(training).glob("train_metrics_rank*.jsonl"):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    maximum = max(maximum, int(json.loads(line)["global_step"]))
                except json.JSONDecodeError:
                    # An interrupted final append cannot establish a completed step.
                    continue
    return maximum


def verify_training_gates(config, config_path, run):
    preflight_path = run / "preflight/preflight_audit.json"
    baseline_path = run / "predictions/base/completed.json"
    preflight, baseline = read_json(preflight_path), read_json(baseline_path)
    assert preflight["passed"] is True, "Complete preprocessing preflight is required"
    assert preflight["records_verified"] == 14543 + 2209
    assert preflight["frozen_config_sha256_at_preflight"] == sha256(config_path)
    for key in ("all_images_preserved", "all_targets_complete", "assistant_only_loss",
                "original_order_and_source_ids_preserved", "no_test_or_second_stage_records_processed"):
        assert preflight[key] is True, key
    assert baseline["passed"] is True, "Frozen base-model validation must finish before training"
    assert baseline["frozen_config_sha256"] == sha256(config_path)
    assert baseline["partition"] == "validation" and baseline["records"] == 2209
    assert baseline["stage"] == "base"
    assert baseline["generation_records"] == baseline["loss_records"] == baseline["expected_records"] == 2209
    assert baseline["test_predictions"] == baseline["second_stage_predictions"] == 0
    assert sha256(baseline["generation"]) == baseline["generation_sha256"]
    assert sha256(baseline["loss"]) == baseline["loss_sha256"]
    return {"preflight": str(preflight_path), "preflight_sha256": sha256(preflight_path),
            "baseline": str(baseline_path), "baseline_sha256": sha256(baseline_path)}


def verify_ordered_dataset(trainer, rows):
    """Check all source messages and image paths before assigning sampler IDs."""
    from swift.llm.dataset import LazyLLMDataset
    assert isinstance(trainer.train_dataset, LazyLLMDataset) and trainer.train_dataset.strict
    assert len(trainer.train_dataset) == len(rows) == 14543
    assert trainer.eval_dataset is None
    dataset = trainer.train_dataset.dataset
    for index, row in enumerate(rows):
        raw = dataset[index]
        messages = [{"role": message["role"], "content": message["content"]}
                    for message in raw["messages"]]
        assert messages == row["messages"], f"Dataset message ordering changed at row{index}"
        paths = [image if isinstance(image, str) else image["path"] for image in raw["images"]]
        assert paths == row["images"], f"Dataset image ordering changed at row{index}"


def train(config_path, run, plan_only=False):
    configure_environment(cpu=plan_only)
    config = load_config(config_path, verify=True)
    assert str(run) == config["paths"]["run"], "Use the frozen run directory"
    assert config["training"]["world_size"] == 2 and config["training"]["effective_batch_size"] == 16
    rows, sidecars, prompts = load_records(config, "initial_sft")
    assert stable_hash([row["id"] for row in sidecars]) == config["data"]["initial_sft"]["ordered_ids_sha256"]
    assert Counter(row["task"] for row in sidecars) == {"explain": 1403, "track": 13140}
    assert all(row["participant"] in {f"P{i:02}" for i in range(1, 7)} for row in sidecars)
    metadata = [{key: row[key] for key in ("id", "task", "participant")} for row in sidecars]
    del sidecars, prompts
    training = run / "training"
    output = training / "checkpoints"
    from .train_audit import (TrainingAuditCallback, install_distributed_loader,
                              select_resume_checkpoint, validate_checkpoint, write_json)
    config_sha = sha256(config_path)
    resume, rejected = select_resume_checkpoint(output, config_sha)
    previous_step = maximum_observed_step(training)
    if not resume and previous_step:
        raise RuntimeError(f"Logs show completed step{previous_step}, but no resumable checkpoint exists; refusing a fresh restart")
    argv = build_swift_argv(config, output, resume)
    packages = {name: importlib.metadata.version(name) for name in
                ("ms-swift", "transformers", "peft", "accelerate", "trl", "torch", "torchvision")}
    assert packages == {"ms-swift": "3.12.6", "transformers": "4.57.1", "peft": "0.17.1",
                        "accelerate": "1.10.1", "trl": "0.23.1", "torch": "2.6.0+cu124",
                        "torchvision": "0.21.0+cu124"}, packages
    invocation = {"argv": argv, "frozen_config_sha256": config_sha, "packages": packages,
                  "original_checkpoint": config["model"]["path"], "revision": config["model"]["revision"],
                  "resume": resume, "rejected_newer_checkpoints": rejected,
                  "maximum_preexisting_logged_step": previous_step,
                  "unsaved_steps_replayed": max(0, previous_step - (resume["global_step"] if resume else 0)),
                  "fresh_start_from_original_model": resume is None,
                  "dataset_records": 14543, "natural_task_counts": dict(Counter(row["task"] for row in metadata)),
                  "validation_or_second_stage_in_training": False, "sampler_padding_is_explicit": True,
                  "plan_only": plan_only,
                  "source_sha256": {str(path): sha256(path) for path in
                                    (Path(__file__), Path(__file__).with_name("train_audit.py"),
                                     Path(__file__).with_name("common.py"))}}
    if plan_only:
        print(json.dumps(invocation, indent=2), flush=True)
        return invocation
    expected_devices = ",".join(GPU_UUIDS[index] for index in (0, 1))
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected_devices
    assert os.environ.get("WORLD_SIZE") == "2" and os.environ.get("LOCAL_RANK") in ("0", "1")
    rank = int(os.environ["RANK"])
    assert rank in (0, 1)
    invocation["gates"] = verify_training_gates(config, config_path, run)
    write_json(training / f"training_invocation_rank{rank}.json", invocation)
    if resume and resume["global_step"] == 1818:
        index = read_json(training / "checkpoint_index.json")
        for epoch in (1, 2):
            snapshot = validate_checkpoint(index[f"epoch{epoch}"], config_sha)
            assert snapshot["global_step"] == epoch * 909
        write_json(training / f"already_complete_rank{rank}.json", {"passed": True, "resume": resume})
        print("Training is already complete at 1,818 steps; no model loaded or optimizer step taken", flush=True)
        return invocation
    assert resume is None or 0 <= resume["global_step"] < 1818
    import torch
    from swift.llm.train.sft import SwiftSft
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert torch.cuda.is_bf16_supported()

    class AuditedSft(SwiftSft):
        def _prepare_template(self):
            super()._prepare_template()
            # Swift's CLI says delete; its template maps this to raise. Assert
            # that behavior so no target truncation or record substitution occurs.
            assert self.template.truncation_strategy == "raise"
            assert self.template.max_length == 8192
            assert self.template.loss_scale.base_strategy == "default"
            assert self.processor.global_vars["image_min_token_num"] == 64
            assert self.processor.global_vars["image_max_token_num"] == 144

        def train(self, trainer):
            assert self.args.model == config["model"]["path"] and not self.args.adapters
            assert not self.args.dataset_shuffle and not self.args.packing and not self.args.padding_free
            assert self.args.val_dataset == [] and self.args.split_dataset_ratio == 0
            assert trainer.args.output_dir == str(output)
            verify_ordered_dataset(trainer, rows)
            schedule = install_distributed_loader(trainer, metadata, seed=config["training"]["data_seed"])
            write_json(training / f"training_schedule_rank{rank}.json", schedule)
            if rank == 0:
                write_json(training / "training_schedule.json", schedule)
            callback = TrainingAuditCallback(training, trainer, metadata, config_sha, schedule, resume=resume)
            callback.install_loss_guard()
            trainer.add_callback(callback)
            result = super().train(trainer)
            assert trainer.state.global_step == 1818
            write_json(training / f"swift_train_result_rank{rank}.json", result)
            return result

    return AuditedSft(argv).main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--plan-only", action="store_true", help="CPU invocation check; load no model")
    args = parser.parse_args()
    train(args.config.resolve(), args.run.resolve(), plan_only=args.plan_only)


if __name__ == "__main__":
    main()
