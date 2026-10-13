"""Read-only environment inventory and a non-executable training proposal."""
import argparse
import importlib.metadata as metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from code.e0.dataset import atomic_write_json, sha256_file
from .revise import DEFAULT_OUTPUT, PROJECT

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_CHECKPOINT = PROJECT / f".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/{CHECKPOINT_REVISION}"

def inventory(out, ck=DEFAULT_CHECKPOINT):
    ck = ck.resolve()
    packages = {"torch", "transformers", "ms-swift", "swift", "trl", "peft", "accelerate", "vllm", "deepspeed", "flash-attn"}
    roots = [PROJECT / ".venvs/hdepic-swift-8b", PROJECT / ".venvs/llavaction", Path("/data/kailaic/envs/qic"), Path("/data/kailaic/envs/vllm"), Path("/data/kailaic/envs/vllm030"), Path("/home/yaxuanli/miniforge3"), Path("/home/yaxuanli/miniforge3/envs/codex")]
    sites = [p for root in roots for p in root.glob("lib/python*/site-packages")] + [PROJECT / ".runtime457", PROJECT / ".runtime_qwen"]
    environments = {}
    for site in sites:
        environments[str(site)] = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in metadata.distributions(path=[str(site)]) if d.metadata.get("Name", "").lower().replace("_", "-") in packages}
    configs = {name: json.loads((ck / name).read_text()) for name in ("config.json", "preprocessor_config.json")}
    assert configs["config.json"]["model_type"] == "qwen3_vl", "Pilot requires the dense Qwen3-VL-8B model"
    weights = json.loads((ck / "model.safetensors.index.json").read_text())
    shards = sorted(set(weights["weight_map"].values()))
    assert all((ck / name).is_file() for name in shards)
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    gpu = {"query_success": query.returncode == 0, "observed_at_utc": datetime.now(timezone.utc).isoformat(), "stdout": query.stdout, "stderr": query.stderr, "returncode": query.returncode}
    saved = out / "gpu_inventory_verified.json"
    if saved.exists():
        gpu["last_successful_host_inspection"] = json.loads(saved.read_text())
    if query.returncode == 0:
        rows = list(__import__("csv").reader(query.stdout.splitlines()))
        gpu["devices"] = [{"index": int(r[0]), "model": r[1].strip(), "total_mib": int(r[2]), "used_mib": int(r[3]), "free_mib": int(r[4]), "utilization_percent": int(r[5])} for r in rows]
    result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": {"model": MODEL_ID, "revision": CHECKPOINT_REVISION, "path": str(ck), "all_weight_shards_present": True, "shard_count": len(shards), "weight_bytes": sum((ck / p).stat().st_size for p in shards), "configurations": configs, "config_sha256": {n: sha256_file(ck / n) for n in configs}}, "environments": environments, "ms_swift_version": next((e["ms-swift"] for e in environments.values() if "ms-swift" in e), None), "gpu": gpu, "model_loaded_during_inventory": False}
    atomic_write_json(out / "local_environment.json", result)
    proposal = {
        "status": "10_step_disposable_smoke_configuration_full_run_pending_memory_measurement",
        "model": MODEL_ID, "model_revision": CHECKPOINT_REVISION, "model_path": str(ck),
        "training_framework": {"name": "ms-swift", "installed_version": result["ms_swift_version"], "environment": str(PROJECT / ".venvs/hdepic-swift-8b")},
        "precision": "bfloat16", "finetuning": "text-model LoRA", "lora_rank": 16, "lora_alpha": 32,
        "freeze_vit": True, "freeze_aligner": True, "freeze_llm": False,
        "device_requirement": "one allocated idle 48 GiB GPU", "device_ids": None,
        "data_parallel_world_size": 1, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1, "effective_batch_examples": 1,
        "max_optimizer_steps": 10, "disposable_checkpoint": True,
        "eventual_full_run_initialization": "original 8B checkpoint; never resume disposable smoke adapter",
        "context_tokens": 8192, "gradient_checkpointing": True, "packing": False,
        "loss_scale": "default", "loss": "assistant answer tokens and assistant terminator only",
        "smoke_sampling": "unique initial-SFT records, both tasks and goal-present/absent; controlled balanced coverage for pipeline validation",
        "full_run_mixture": "natural mixture unless a later explicitly recorded sampler is selected",
        "images_per_record": 16, "image_min_pixels": 65536, "image_max_pixels": 147456,
        "image_patch_size": 16, "image_spatial_merge_size": 2,
        "max_visual_tokens_per_image": 144, "max_visual_tokens_16_images": 2304,
        "truncate_policy": "fail if any image or any assistant answer token is removed",
        "full_run_configuration_selection": "pending measured single-GPU peak memory and step time",
        "second_stage_policy": "WHY-only extra-SFT and GRPO use identical second_stage_explain_ids.json; no second-stage records enter smoke",
        "test_predictions": "prohibited during this task",
    }
    atomic_write_json(out / "proposed_training_config.json", proposal)
    print(f"Environment/config inspection saved; no model loaded. ms-swift installed version: {result['ms_swift_version']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    inventory(p.parse_args().output.resolve())
