"""CPU-only local tokenizer/image preflight; never loads model weights."""
import argparse
import json
from pathlib import Path

from code.e0.dataset import atomic_write_json


def structured(item):
    images = iter(item["images"])
    messages = []
    for m in item["messages"]:
        content = []
        parts = m["content"].split("<image>")
        for i, text in enumerate(parts):
            if text:
                content.append({"type": "text", "text": text})
            if i + 1 < len(parts):
                content.append({"type": "image", "image": next(images)})
        messages.append({"role": m["role"], "content": content})
    assert next(images, None) is None
    return messages


def preflight(out, checkpoint):
    from transformers import AutoProcessor
    import transformers
    import torch
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model_config = json.loads((checkpoint / "config.json").read_text())
    image_id = model_config["image_token_id"]
    export = [json.loads(l) for l in (out / "sample/training20.jsonl").open()]
    scoring = [json.loads(l) for l in (out / "sample/training20.scoring.jsonl").open()]
    rows = []
    for item, s in zip(export, scoring):
        messages = structured(item)
        # This uses the local HF template; ms-swift's actual assistant loss mask
        # and training preprocessing still require the future smoke test.
        kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"images_kwargs": {"min_pixels": 65536, "max_pixels": 147456}})
        full = processor.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
        prompt = processor.apply_chat_template(messages[:2], add_generation_prompt=True, **kwargs)
        ids, prefix = full["input_ids"][0], prompt["input_ids"][0]
        grid = full["image_grid_thw"]
        assert len(grid) == len(item["images"]) == 16
        visual = int((grid.prod(dim=1) // processor.image_processor.merge_size**2).sum())
        assert int((grid.prod(dim=1) // processor.image_processor.merge_size**2).max()) <= 144
        assert int((ids == image_id).sum()) == visual
        assert len(ids) <= 8192 and torch.equal(ids[:len(prefix)], prefix)
        assistant = processor.tokenizer.decode(ids[len(prefix):], skip_special_tokens=False)
        assert item["messages"][2]["content"] in assistant
        rows.append({"id": s["id"], "task": s["task"], "goal_present": s["goal_status"] == "available", "individual_images_after_preprocessing": len(grid), "image_grid_thw": grid.tolist(), "visual_tokens": visual, "full_context_tokens": len(ids), "prompt_tokens": len(prefix), "assistant_suffix_tokens": len(ids) - len(prefix), "complete_target_survives": True})
        print(f"CPU processor verified {s['id']}: 16 images, {len(ids)} tokens", flush=True)
    assert len(rows) == 20
    atomic_write_json(out / "processor_preflight.json", {"passed": True, "records_verified": 20, "transformers_version": transformers.__version__, "torch_version": torch.__version__, "processor_class": type(processor).__name__, "processor_checkpoint": str(checkpoint), "image_min_pixels": 65536, "image_max_pixels": 147456, "context_limit": 8192, "model_weights_loaded": False, "model_inference_calls": 0, "training_started": False, "rows": rows, "limitations": "CPU HF processor/template preflight only; no ms-swift loss-mask check, gradients, LoRA updates, checkpoint reload, or model-quality validation."})


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=project / "datasets/hdepic_sft_v2")
    p.add_argument("--checkpoint", type=Path, default=project / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    a = p.parse_args(); preflight(a.output.resolve(), a.checkpoint.resolve())
