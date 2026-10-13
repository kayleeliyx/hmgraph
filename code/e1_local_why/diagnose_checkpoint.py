"""Minimal untouched-runtime checkpoint sanity check (not an evaluation)."""

from __future__ import annotations

import os
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


CHECKPOINT = Path(
    "/data/yaxuanli/ActGround/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3,4":
        raise RuntimeError("diagnostic must expose physical GPUs 3 and 4")
    processor = AutoProcessor.from_pretrained(
        CHECKPOINT, local_files_only=True, trust_remote_code=False
    )
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        CHECKPOINT,
        local_files_only=True,
        trust_remote_code=False,
        device_map="balanced",
        max_memory={0: "38GiB", 1: "38GiB"},
        dtype="auto",
        low_cpu_mem_usage=True,
    ).train()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]

    def checkpoint_values(name: str, selection: tuple[object, ...]) -> list[float]:
        with safe_open(CHECKPOINT / index[name], framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[selection].float().tolist()

    language_model = model.model.language_model
    comparisons = [
        (
            "model.language_model.embed_tokens.weight",
            language_model.embed_tokens.weight[0, :8].detach().float().cpu().tolist(),
            (0, slice(0, 8)),
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            language_model.layers[0].self_attn.q_proj.weight[0, :8]
            .detach()
            .float()
            .cpu()
            .tolist(),
            (0, slice(0, 8)),
        ),
        (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            language_model.layers[0].mlp.experts.gate_up_proj[0, 0, :8]
            .detach()
            .float()
            .cpu()
            .tolist(),
            (0, 0, slice(0, 8)),
        ),
        (
            "lm_head.weight",
            model.lm_head.weight[0, :8].detach().float().cpu().tolist(),
            (0, slice(0, 8)),
        ),
    ]
    for name, loaded, selection in comparisons:
        saved = checkpoint_values(name, selection)
        print("weight_match", name, loaded == saved, loaded, saved)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who are you?"}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
        finite = torch.isfinite(logits)
        top_values, top_ids = torch.topk(torch.nan_to_num(logits, nan=-float("inf")), 5)
        print(
            "logits",
            {
                "finite": int(finite.sum()),
                "total": logits.numel(),
                "nan": int(torch.isnan(logits).sum()),
                "inf": int(torch.isinf(logits).sum()),
                "top_ids": top_ids.tolist(),
                "top_values": top_values.tolist(),
            },
        )
        output = model.generate(
            **inputs, max_new_tokens=8, do_sample=False, use_cache=True
        )
    new_ids = output[0, inputs["input_ids"].shape[-1] :]
    print("generated_ids", new_ids.tolist())
    print(
        "decoded",
        repr(
            processor.decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ),
    )


if __name__ == "__main__":
    main()
