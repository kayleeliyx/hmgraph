"""Disposable 10-step ms-swift smoke on an explicitly allocated idle GPU.

No validation/test rows, full SFT, GRPO, or remote inference. Always initialize
training from the pinned original dense 8B snapshot; never resume this adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from code.e0.dataset import atomic_write_json, atomic_write_jsonl, sha256_file

PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "datasets/hdepic_sft_v2_repaired"
RUN = PROJECT / "runs/hdepic_sft_smoke_8b"
MODEL = PROJECT / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def read_rows(p):
    return [json.loads(l) for l in p.open() if l.strip()]


def local_environment():
    # Set before importing Swift/Qwen utilities. The pinned local snapshot and
    # offline data caches are shared by preprocessing, training, and reload.
    for key, value in {
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "USE_HF": "1",
        "HF_HOME": str(PROJECT / ".cache/huggingface"),
        "HF_DATASETS_CACHE": str(PROJECT / ".cache/hdepic_swift_dataset"),
        "MODELSCOPE_CACHE": str(PROJECT / ".cache/modelscope"),
        "SWIFT_CACHE_DIR": str(PROJECT / ".cache/swift"),
        "IMAGE_MAX_TOKEN_NUM": "144", "IMAGE_MIN_TOKEN_NUM": "64",
        "OMP_NUM_THREADS": "4",
    }.items():
        os.environ[key] = value


def prepare(dataset, run):
    exports = read_rows(dataset / "sample/training20.jsonl")
    scoring = read_rows(dataset / "sample/training20.scoring.jsonl")
    assert len(exports) == len(scoring) == 20
    categories = [("explain", True), ("track", True), ("explain", False), ("track", False)]
    choices = {key: [(r, s) for r, s in zip(exports, scoring) if s["task"] == key[0] and (s["goal_status"] == "available") == key[1]] for key in categories}
    chosen = [choices[key][i] for i in range(2) for key in categories]
    rows, sidecars = [], []
    for r, s in chosen:
        assert s["partition"] == "initial_sft" and s["participant"] in {f"P{i:02}" for i in range(1, 7)}
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert len(r["images"]) == r["messages"][1]["content"].count("<image>") == 16
        rows.append(r); sidecars.append({**s, "row_index": len(rows) - 1})
    assert len({s["id"] for s in sidecars}) == len(rows) == 8
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(run / "smoke_initial8.jsonl", rows)
    atomic_write_jsonl(run / "smoke_initial8.scoring.jsonl", sidecars)
    atomic_write_json(run / "smoke_dataset_manifest.json", {"original_model": str(MODEL), "dataset": str(dataset), "ids": [s["id"] for s in sidecars], "partition": "initial_sft", "records": 8, "duplicated_file_rows": 0, "task_goal_categories": {f"{t}:goal_{g}": 2 for t, g in categories}, "ordering": "interleaved task/goal categories; repeated epochs are explicit under max_steps=10", "natural_mixture_claim": False, "purpose": "pipeline smoke coverage only", "split_sha256": sha256_file(dataset / "split_manifest.json"), "dataset_sha256": sha256_file(run / "smoke_initial8.jsonl")})
    print("Prepared 8 unique initial-SFT rows, two per task/goal category", flush=True)


def gpu_guard(run, phase):
    allocation = json.loads((run / "gpu_allocation.json").read_text())
    assert allocation.get("authorized_by_user") is True, "GPU allocation is required"
    uuid = allocation["gpu_uuid"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "Use only the allocated physical GPU UUID"
    result = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    row = next(csv.reader(result.stdout.splitlines()))
    used, free, utilization = map(int, (row[4], row[5], row[6]))
    processes = subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    assert used < 1024 and free >= 47000 and utilization == 0 and not processes.stdout.strip(), "Allocated GPU is occupied: do not start"
    observed = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_index": int(row[0]), "gpu_uuid": row[1].strip(), "name": row[2].strip(), "total_mib": int(row[3]), "used_mib": used, "free_mib": free, "utilization_percent": utilization, "compute_processes": [], "authorized_by_user": True}
    atomic_write_json(run / f"gpu_before_{phase}_load.json", observed)
    return observed


def train(run):
    # The guard runs before importing/loading a model. Allocation cannot be
    # inferred from an idle device or from a stale occupancy snapshot.
    local_environment()
    gpu_guard(run, "train")
    import torch
    from swift.llm.train.sft import SwiftSft
    from .swift_smoke_audit import SmokeAuditCallback, audit_training_preprocessing
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    output = run / "disposable"
    assert not (output / "checkpoint-10").exists(), "Use a fresh run directory; never resume the smoke checkpoint"
    rows = read_rows(run / "smoke_initial8.jsonl")
    sidecars = read_rows(run / "smoke_initial8.scoring.jsonl")
    argv = ["--model", str(MODEL), "--model_type", "qwen3_vl", "--use_hf", "true", "--train_type", "lora", "--tuner_backend", "peft", "--target_modules", "all-linear", "--freeze_vit", "true", "--freeze_aligner", "true", "--freeze_llm", "false", "--lora_rank", "16", "--lora_alpha", "32", "--lora_dropout", "0", "--torch_dtype", "bfloat16", "--bf16", "true", "--attn_impl", "sdpa", "--dataset", str(run / "smoke_initial8.jsonl"), "--split_dataset_ratio", "0", "--dataset_shuffle", "false", "--train_dataloader_shuffle", "false", "--dataset_num_proc", "1", "--dataloader_num_workers", "0", "--lazy_tokenize", "true", "--strict", "true", "--max_length", "8192", "--max_pixels", "147456", "--truncation_strategy", "delete", "--loss_scale", "default", "--padding_free", "false", "--packing", "false", "--gradient_checkpointing", "true", "--gradient_checkpointing_kwargs", '{"use_reentrant":false}', "--vit_gradient_checkpointing", "false", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1", "--max_steps", "10", "--learning_rate", "0.0001", "--lr_scheduler_type", "constant", "--warmup_steps", "0", "--optim", "adamw_torch", "--logging_steps", "1", "--logging_nan_inf_filter", "false", "--eval_strategy", "no", "--save_strategy", "steps", "--save_steps", "10", "--save_total_limit", "1", "--save_only_model", "true", "--save_safetensors", "true", "--output_dir", str(output), "--add_version", "false", "--report_to", "none", "--seed", "20261003", "--data_seed", "20261003", "--push_to_hub", "false"]
    atomic_write_json(run / "smoke_train_invocation.json", {"argv": argv, "original_checkpoint": str(MODEL), "resume_from_checkpoint": None, "max_optimizer_steps": 10, "disposable": True, "full_training_launched": False, "grpo_launched": False, "test_predictions": 0})

    class AuditSft(SwiftSft):
        def train(self, trainer):
            audit_training_preprocessing(trainer, rows, sidecars, run)
            callback = SmokeAuditCallback(run)
            trainer.add_callback(callback)
            result = super().train(trainer)
            assert trainer.state.global_step == 10
            atomic_write_json(run / "swift_train_result.json", result)
            return result

    AuditSft(argv).main()
    checkpoint = output / "checkpoint-10"
    assert (checkpoint / "adapter_config.json").is_file() and (checkpoint / "adapter_model.safetensors").is_file()
    atomic_write_json(run / "saved_checkpoint_identity.json", {"checkpoint": str(checkpoint), "disposable": True, "optimizer_steps": 10, "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in checkpoint.iterdir() if p.is_file()}, "full_run_must_start_from": str(MODEL)})
    print("Finished exactly 10 optimizer steps; disposable checkpoint saved", flush=True)


def reload_and_generate(run):
    local_environment()
    gpu_guard(run, "reload")
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from swift.llm import InferArguments, InferRequest, RequestConfig
    from swift.llm.infer.infer import SwiftInfer
    torch.set_num_threads(4)
    checkpoint = run / "disposable/checkpoint-10"
    saved = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    args = InferArguments(model=str(MODEL), model_type="qwen3_vl", adapters=[str(checkpoint)], infer_backend="pt", torch_dtype="bfloat16", attn_impl="sdpa", use_hf=True, load_args=False, max_batch_size=1, max_length=8192, max_pixels=147456)
    inference = SwiftInfer(args)
    loaded = get_peft_model_state_dict(inference.infer_engine.model)
    assert set(saved) == set(loaded)
    for key in saved:
        assert saved[key].shape == loaded[key].shape and torch.equal(saved[key], loaded[key].detach().cpu()), key
    atomic_write_json(run / "checkpoint_reload_audit.json", {"passed": True, "fresh_process": True, "base_model": str(MODEL), "adapter": str(checkpoint), "exact_saved_adapter_tensors_reloaded": len(saved), "no_merge_or_answer_prefix_forcing": True})
    items, sidecars = read_rows(run / "smoke_initial8.jsonl"), read_rows(run / "smoke_initial8.scoring.jsonl")
    generations = []
    for item, s in list(zip(items, sidecars))[:4]:
        assert s["partition"] == "initial_sft"
        request = InferRequest(messages=item["messages"][:2], images=item["images"])
        response = inference.infer([request], RequestConfig(max_tokens=256, temperature=0), use_tqdm=False)[0]
        text = response.choices[0].message.content
        tags = re.findall(r"<([a-z]+)>", text)
        expected = ["why" if s["task"] == "explain" else "action"] + (["goal"] if s["goal_status"] == "available" else []) + ["next"]
        balanced = all(len(re.findall(f"<{tag}>.*?</{tag}>", text, re.S)) == 1 for tag in expected)
        next_text = re.search(r"<next>(.*?)</next>", text, re.S)
        complete_tag_pattern = r"\s*" + r"\s*".join(f"<{tag}>.+?</{tag}>" for tag in expected) + r"\s*"
        tags_only = re.fullmatch(complete_tag_pattern, text, re.S) is not None
        next_actions = [v.strip() for v in next_text.group(1).split(";")] if next_text else []
        generations.append({"id": s["id"], "partition": s["partition"], "task": s["task"], "goal_status": s["goal_status"], "input_images": len(item["images"]), "generation": text, "expected_tags": expected, "observed_open_tags": tags, "tag_format_valid": tags == expected and balanced and tags_only and len(next_actions) == 3 and all(next_actions), "tags_only_no_surrounding_prose": tags_only, "semicolon_next_action_count": len(next_actions), "finish_reason": response.choices[0].finish_reason, "quality_claim": "none; diagnostic training-row generations only"})
        print(f"Reloaded generation {s['task']} goal={s['goal_status']}: {text}", flush=True)
    atomic_write_jsonl(run / "reload_initial_generations.jsonl", generations)
    atomic_write_json(run / "generation_audit.json", {"records": 4, "only_initial_sft": True, "tag_format_passed": sum(r["tag_format_valid"] for r in generations), "format_failures_reported_without_forcing": True, "test_predictions": 0, "quality_claim": "No model-quality conclusion from this smoke test"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "train", "reload"), required=True)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--run", type=Path, default=RUN)
    a = p.parse_args()
    if a.phase == "prepare":
        prepare(a.dataset.resolve(), a.run.resolve())
    elif a.phase == "train":
        train(a.run.resolve())
    else:
        reload_and_generate(a.run.resolve())


if __name__ == "__main__":
    main()
