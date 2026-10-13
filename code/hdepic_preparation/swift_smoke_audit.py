"""Audits for the disposable ms-swift Qwen3-VL-8B smoke test.

No function in this module starts training, inference, or checkpoint loading.
The harness owns GPU allocation and the actual SwiftSft/SwiftInfer lifecycles.
Validated against ms-swift 3.12.6 and Transformers 4.57.1 APIs.
"""

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import TrainerCallback


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _tensor_hash(tensor):
    """Hash exact tensor bytes, including dtypes NumPy does not support."""
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_1d(value):
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    assert tensor.ndim in (1, 2)
    if tensor.ndim == 2:
        assert tensor.shape[0] == 1, tensor.shape
    return tensor.reshape(-1)


def audit_collated_record(batch, row, scoring, template, max_length=None):
    """Audit one real Swift trainer collator output against its original row."""
    assert scoring["partition"] == "initial_sft", scoring["id"]
    assert scoring["participant"] in {f"P{i:02d}" for i in range(1, 7)}
    messages = row["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert all(isinstance(message["content"], str) for message in messages)
    images = row["images"]
    assert len(images) == len(set(images)) == 16
    assert messages[1]["content"].count("<image>") == 16
    assert all(Path(path).is_file() for path in images)
    source_frames = scoring["frames"]
    assert [frame["path"] for frame in source_frames] == images
    frame_times = [frame["timestamp_sec"] for frame in source_frames]
    assert frame_times == sorted(frame_times)
    assert all(time_sec <= scoring["query_time_sec"] for time_sec in frame_times)

    tokenizer = template.tokenizer
    processor = template.processor
    ids, labels = _tensor_1d(batch["input_ids"]), _tensor_1d(batch["labels"])
    assert len(ids) == len(labels)
    limit = max_length if max_length is not None else template.max_length
    assert len(ids) <= limit, (scoring["id"], len(ids), limit)
    if "attention_mask" in batch:
        assert _tensor_1d(batch["attention_mask"]).bool().all().item(), "Unexpected padding in batch size 1"
    grid = torch.as_tensor(batch["image_grid_thw"]).detach().cpu()
    assert tuple(grid.shape) == (16, 3), grid.shape
    merge_size = processor.image_processor.merge_size
    per_image_tokens = grid.prod(dim=1) // merge_size**2
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_mask = ids == image_id
    assert int((ids == vision_start).sum()) == int((ids == vision_end).sum()) == 16
    assert int(image_mask.sum()) == int(per_image_tokens.sum())
    assert (labels[image_mask] == -100).all().item()
    assert (labels[(ids == vision_start) | (ids == vision_end)] == -100).all().item()
    pixels = batch["pixel_values"]
    assert pixels.shape[0] == int(grid.prod(dim=1).sum()), pixels.shape

    # Image-group order is preserved by the actual template/collator. Check
    # each group individually, rather than just its aggregate token count.
    starts = (ids == vision_start).nonzero().reshape(-1).tolist()
    ends = (ids == vision_end).nonzero().reshape(-1).tolist()
    for start, end, token_count in zip(starts, ends, per_image_tokens.tolist()):
        assert end - start - 1 == token_count
        assert (ids[start + 1:end] == image_id).all().item()

    supervised = labels != -100
    assert supervised.any().item()
    assert torch.equal(labels[supervised], ids[supervised])
    supervised_positions = supervised.nonzero().reshape(-1)
    first_answer = int(supervised_positions[0])
    assert torch.equal(supervised_positions, torch.arange(first_answer, len(ids)))
    assert (labels[:first_answer] == -100).all().item()
    decoded_answer = tokenizer.decode(labels[supervised].tolist(), skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
    target = messages[-1]["content"]
    # Qwen3-VL's Swift ChatML suffix includes EOT plus a newline. Permit only
    # that exact assistant terminator; never arbitrary text around the target.
    terminator = decoded_answer[len(target):] if decoded_answer.startswith(target) else None
    assert terminator in ("<|im_end|>", "<|im_end|>\n"), (scoring["id"], decoded_answer, target)

    full_text = tokenizer.decode(ids.tolist(), skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
    collapsed = re.sub(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>", "<image>", full_text)
    expected_prompt = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages[:2]) + "<|im_start|>assistant\n"
    assert collapsed == expected_prompt + target + terminator, scoring["id"]
    decoded_masked = tokenizer.decode(ids[:first_answer].tolist(), skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
    collapsed_masked = re.sub(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>", "<image>", decoded_masked)
    assert collapsed_masked == expected_prompt, "Assistant header must also be masked"
    goal_present = "<goal>" in target
    assert goal_present == (scoring["goal_status"] == "available")
    primary_tag = "why" if scoring["task"] == "explain" else "action"
    assert target.startswith(f"<{primary_tag}>") and f"</{primary_tag}>" in target
    assert target.endswith("</next>")
    return {
        "id": scoring["id"], "participant": scoring["participant"],
        "partition": "initial_sft", "task": scoring["task"],
        "goal_present": goal_present, "goal_status": scoring["goal_status"],
        "ordered_image_paths": images, "ordered_source_timestamps_sec": frame_times,
        "individual_images_after_collator": 16, "image_grid_thw": grid.tolist(),
        "image_tokens_per_image": per_image_tokens.tolist(), "image_tokens": int(image_mask.sum()),
        "pixel_values_shape": list(pixels.shape), "input_tokens": len(ids),
        "masked_prompt_tokens_including_assistant_header": first_answer,
        "supervised_tokens_including_assistant_terminator": int(supervised.sum()),
        "decoded_supervised_tokens": decoded_answer,
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "permitted_supervised_assistant_terminator": terminator,
        "complete_target": True, "assistant_only_loss": True,
        "system_user_image_and_assistant_header_masked": True,
    }


def audit_training_preprocessing(trainer, source_rows, scoring_rows, output_dir, expected_records=8):
    """Inspect the real trainer's encoded dataset and batch-size-one collator.

    The harness must disable dataset shuffling so source/scoring row order
    matches the trainer dataset. A byte-exact full prompt/target comparison
    detects wrong ordering or silently substituted examples regardless.
    Returns a JSON-serializable summary and writes preprocessing_audit.json.
    """
    assert len(source_rows) == len(scoring_rows) == len(trainer.train_dataset) == expected_records
    assert len({row["id"] for row in scoring_rows}) == expected_records
    assert trainer.args.per_device_train_batch_size == 1
    assert trainer.template.mode == "train"
    assert trainer.template.loss_scale.base_strategy == "default"
    audited = []
    for index, (row, scoring) in enumerate(zip(source_rows, scoring_rows)):
        encoded = trainer.train_dataset[index]
        batch = trainer.data_collator([encoded])
        audited.append(audit_collated_record(batch, row, scoring, trainer.template))
    groups = Counter((row["task"], row["goal_present"]) for row in audited)
    assert set(groups) == {(task, goal) for task in ("explain", "track") for goal in (True, False)}
    assert all(count >= 1 for count in groups.values())
    summary = {
        "passed": True, "records_verified": len(audited),
        "pipeline": "actual ms-swift trainer.train_dataset encoding and trainer.data_collator",
        "trainer_class": f"{type(trainer).__module__}.{type(trainer).__name__}",
        "template_class": f"{type(trainer.template).__module__}.{type(trainer.template).__name__}",
        "loss_scale": trainer.template.loss_scale.base_strategy,
        "all_images_preserved": True, "all_targets_complete": True,
        "assistant_only_loss": True, "context_limit": trainer.template.max_length,
        "max_observed_input_tokens": max(row["input_tokens"] for row in audited),
        "task_goal_counts": {f"{task}:goal_{'present' if goal else 'absent'}": count
                             for (task, goal), count in sorted(groups.items())},
        "rows": audited,
    }
    _write_json(Path(output_dir) / "preprocessing_audit.json", summary)
    return summary


class SmokeAuditCallback(TrainerCallback):
    """Check ten real optimizer steps without copying the frozen base weights."""

    def __init__(self, output_dir, expected_steps=10):
        self.output_dir = Path(output_dir)
        self.expected_steps = expected_steps
        self.before = {}
        self.before_hashes = {}
        self.gradients, self.steps, self.losses, self.saves = [], [], [], []
        self.gradient_audit_seconds = 0.

    def _write(self, name, value):
        _write_json(self.output_dir / name, value)

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        assert state.global_step == 0, "Smoke must start from the original base without resumed optimizer"
        assert args.bf16
        assert args.per_device_train_batch_size == 1
        assert args.max_steps == self.expected_steps
        assert not args.logging_nan_inf_filter, "Loss logging must expose NaN/Inf rather than filtering them"
        assert bool(getattr(model, "is_gradient_checkpointing", False)), "Gradient checkpointing was not enabled"
        # SwiftMixin enables model checkpointing itself, then temporarily sets
        # args.gradient_checkpointing=False to prevent HF Trainer from also
        # enabling it on the frozen vision tower. Inspect the actual modules.
        checkpointing_modules = {name: bool(module.gradient_checkpointing)
                                 for name, module in model.named_modules()
                                 if hasattr(module, "gradient_checkpointing")}
        text_checkpointing = {name: enabled for name, enabled in checkpointing_modules.items()
                              if "language_model" in name}
        vision_checkpointing = {name: enabled for name, enabled in checkpointing_modules.items()
                                if ".visual" in name}
        assert text_checkpointing and any(text_checkpointing.values()), "Text gradient checkpointing is disabled"
        assert not any(vision_checkpointing.values()), "Frozen vision checkpointing should remain disabled"
        named = dict(model.named_parameters())
        trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
        assert trainable, "No trainable LoRA parameters"
        for name in trainable:
            assert ".lora_A." in name or ".lora_B." in name, name
            assert "language_model" in name or ".lm_head." in name, name
            assert not any(part in name for part in ("visual", "vision_tower", "merger", "aligner", "projector")), name
        assert optimizer is not None
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert optimizer_ids == {id(parameter) for parameter in trainable.values()}
        adapter_config = model.peft_config["default"]
        assert adapter_config.r == 16 and adapter_config.lora_alpha == 32
        self.before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
        self.before_hashes = {name: _tensor_hash(parameter) for name, parameter in self.before.items()}
        self._write("trainable_parameters.json", {
            "tensor_count": len(trainable), "parameter_count": sum(parameter.numel() for parameter in trainable.values()),
            "names": list(trainable), "shapes": {name: list(parameter.shape) for name, parameter in trainable.items()},
            "dtypes": sorted({str(parameter.dtype) for parameter in trainable.values()}),
            "lora_rank": 16, "lora_alpha": 32, "before_sha256": self.before_hashes,
            "vision_and_aligner_trainable_parameters": 0, "optimizer_contains_only_text_lora": True,
            "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
            "trainer_gradient_checkpointing_flag_during_swift_loop": args.gradient_checkpointing,
            "gradient_checkpointing_modules": checkpointing_modules,
            "model_reports_gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
            "bf16_training": args.bf16, "per_device_batch_size": 1,
        })
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.training_started = time.perf_counter()
        self.start_allocated = torch.cuda.memory_allocated()
        self.start_reserved = torch.cuda.memory_reserved()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_started = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        started = time.perf_counter()
        norm_squared, finite_count, nonzero_count = 0., 0, 0
        missing, nonfinite = [], []
        group_counts = {"lora_A": {"finite": 0, "nonzero": 0}, "lora_B": {"finite": 0, "nonzero": 0}}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None, f"Frozen parameter acquired a gradient: {name}"
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            gradient = parameter.grad.detach().float()
            if not torch.isfinite(gradient).all().item():
                nonfinite.append(name)
                continue
            norm = torch.linalg.vector_norm(gradient).item()
            if not math.isfinite(norm):
                nonfinite.append(name)
                continue
            finite_count += 1
            nonzero_count += norm > 0
            norm_squared += norm * norm
            group = "lora_A" if ".lora_A." in name else "lora_B"
            group_counts[group]["finite"] += 1
            group_counts[group]["nonzero"] += norm > 0
        row = {"optimizer_step": state.global_step + 1, "finite_gradient_tensors": finite_count,
               "nonzero_gradient_tensors": nonzero_count, "missing_gradient_names": missing,
               "nonfinite_gradient_names": nonfinite, "gradient_l2_after_clipping": math.sqrt(norm_squared),
               "groups": group_counts}
        self.gradients.append(row)
        self._write("gradient_audit.json", self.gradients)
        self.gradient_audit_seconds += time.perf_counter() - started
        assert not nonfinite, nonfinite
        assert finite_count > 0 and nonzero_count > 0, row

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.steps.append({"optimizer_step": state.global_step,
                           "seconds_including_gradient_audit": time.perf_counter() - self.step_started})
        self._write("optimizer_step_timing.json", self.steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        grad_norm = float(logs["grad_norm"]) if "grad_norm" in logs else None
        self.losses.append({"optimizer_step": state.global_step, "loss": loss,
                            "reported_gradient_norm_before_clipping": grad_norm})
        self._write("loss_audit.json", self.losses)
        assert math.isfinite(loss), logs
        assert grad_norm is None or math.isfinite(grad_norm), logs

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter = checkpoint / "adapter_model.safetensors"
        config = checkpoint / "adapter_config.json"
        self.saves.append({"optimizer_step": state.global_step, "checkpoint": str(checkpoint),
                           "adapter_weights_saved": adapter.is_file(), "adapter_config_saved": config.is_file(),
                           "adapter_bytes": adapter.stat().st_size if adapter.is_file() else None})
        self._write("checkpoint_save_audit.json", self.saves)
        assert adapter.is_file() and config.is_file(), checkpoint

    def on_train_end(self, args, state, control, model=None, **kwargs):
        torch.cuda.synchronize()
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        training_seconds = time.perf_counter() - self.training_started
        changes = []
        for name, parameter in model.named_parameters():
            if name not in self.before:
                continue
            current = parameter.detach().cpu().contiguous()
            assert torch.isfinite(current).all().item(), name
            delta = current.float() - self.before[name].float()
            changes.append({"name": name, "group": "lora_A" if ".lora_A." in name else "lora_B",
                            "shape": list(current.shape), "dtype": str(current.dtype),
                            "before_sha256": self.before_hashes[name], "after_sha256": _tensor_hash(current),
                            "changed_elements": int((delta != 0).sum()),
                            "delta_l2": torch.linalg.vector_norm(delta).item(),
                            "max_abs_delta": delta.abs().max().item()})
        self._write("lora_parameter_delta_audit.json", changes)
        assert state.global_step == self.expected_steps
        assert len(self.gradients) == len(self.steps) == self.expected_steps
        assert len({row["optimizer_step"] for row in self.gradients}) == self.expected_steps
        assert {row["optimizer_step"] for row in self.losses} == set(range(1, self.expected_steps + 1))
        assert all(math.isfinite(row["loss"]) for row in self.losses)
        assert any(row["changed_elements"] for row in changes), "No LoRA parameter changed"
        assert any(row["optimizer_step"] == self.expected_steps for row in self.saves), "Final checkpoint not saved"
        updated = Counter(row["group"] for row in changes if row["changed_elements"])
        self._write("training_audit_summary.json", {
            "passed": True, "optimizer_steps": state.global_step, "finite_logged_losses": True,
            "finite_gradients_each_optimizer_step": True, "nonzero_gradients_each_optimizer_step": True,
            "text_lora_only": True, "vision_and_aligner_frozen": True,
            "trainable_tensor_count": len(changes), "updated_tensor_counts": dict(updated),
            "updated_elements": sum(row["changed_elements"] for row in changes),
            "peak_cuda_allocated_bytes": peak_allocated, "peak_cuda_reserved_bytes": peak_reserved,
            "start_cuda_allocated_bytes": self.start_allocated, "start_cuda_reserved_bytes": self.start_reserved,
            "mean_seconds_per_optimizer_step_including_gradient_audit":
                sum(row["seconds_including_gradient_audit"] for row in self.steps) / len(self.steps),
            "optimizer_step_seconds": [row["seconds_including_gradient_audit"] for row in self.steps],
            "gradient_audit_overhead_seconds": self.gradient_audit_seconds,
            "training_seconds_including_audits_logging_and_checkpoint": training_seconds,
            "checkpoints": self.saves, "checkpoint_is_disposable": True,
            "future_full_sft_must_start_from_original_8b_checkpoint": True,
            "checkpoint_reload_and_tagged_generation": "checked separately by the harness",
        })
        self.before.clear()
