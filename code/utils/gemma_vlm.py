"""Multimodal loading and inference for W window frame prompts.

Inputs:
  Model id/path, sampled frame paths, and L1 prompt string.
Outputs:
  Verbatim decoded model text for one W record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
import transformers
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


class GemmaVLM:
    """Thin plain-transformers wrapper for Qwen3-VL multimodal inference."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
    ) -> None:
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        dtype = getattr(torch, torch_dtype) if self.device != "cpu" else torch.float32
        print(f"[GemmaVLM] transformers_version={transformers.__version__}")
        print(f"[GemmaVLM] model_id={model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        device_map = "auto" if self.device == "auto" else ({"": self.device} if self.device != "cpu" else None)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def _generate_from_content(self, content: list[dict]) -> str:
        """Run Qwen3-VL on multimodal chat content and return verbatim output text."""
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output_ids[:, prompt_len:]
        return self.processor.decode(
            generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def infer(self, frame_paths: Iterable[str | Path], prompt: str) -> str:
        """Run Qwen3-VL on a list of frame paths and return verbatim output text."""
        images: List[Image.Image] = [
            Image.open(path).convert("RGB") for path in frame_paths
        ]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return self._generate_from_content(content)

    def infer_context(
        self,
        before_frame_paths: Iterable[str | Path],
        target_frame_paths: Iterable[str | Path],
        after_frame_paths: Iterable[str | Path],
        prompt: str,
    ) -> str:
        """Run Qwen3-VL on before/target/after W frame groups and return raw text."""
        before_images = [Image.open(path).convert("RGB") for path in before_frame_paths]
        target_images = [Image.open(path).convert("RGB") for path in target_frame_paths]
        after_images = [Image.open(path).convert("RGB") for path in after_frame_paths]
        n_before = len(before_images)
        n_target = len(target_images)
        n_after = len(after_images)
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"The first {n_before} images are BEFORE CONTEXT and are visually labeled "
                    "BEFORE CONTEXT - DO NOT LABEL.\n"
                    f"The next {n_target} images are TARGET and are visually labeled TARGET FRAME "
                    "- LABEL THIS INTERVAL.\n"
                    f"The final {n_after} images are AFTER CONTEXT and are visually labeled "
                    "AFTER CONTEXT - DO NOT LABEL.\n"
                    "Only output actions visible in TARGET frames."
                ),
            }
        ]
        content.extend({"type": "image", "image": image} for image in before_images)
        content.extend({"type": "image", "image": image} for image in target_images)
        content.extend({"type": "image", "image": image} for image in after_images)
        content.append({"type": "text", "text": prompt})
        try:
            return self._generate_from_content(content)
        except Exception:
            fallback_prompt = (
                f"The first {n_before} images are BEFORE CONTEXT and are visually labeled "
                "BEFORE CONTEXT - DO NOT LABEL.\n"
                f"The next {n_target} images are TARGET and are visually labeled TARGET FRAME "
                "- LABEL THIS INTERVAL.\n"
                f"The final {n_after} images are AFTER CONTEXT and are visually labeled "
                "AFTER CONTEXT - DO NOT LABEL.\n"
                "Only output actions visible in TARGET frames.\n\n"
                + prompt
            )
            images = before_images + target_images + after_images
            fallback_content = [{"type": "image", "image": image} for image in images]
            fallback_content.append({"type": "text", "text": fallback_prompt})
            return self._generate_from_content(fallback_content)

    def infer_context_with_crops(
        self,
        before_frame_paths: Iterable[str | Path],
        target_frame_paths: Iterable[str | Path],
        crop_frame_paths: Iterable[str | Path],
        prompt: str,
    ) -> str:
        """Run Qwen3-VL on before/target full frames plus target-only detail crops."""
        before_images = [Image.open(path).convert("RGB") for path in before_frame_paths]
        target_images = [Image.open(path).convert("RGB") for path in target_frame_paths]
        crop_images = [Image.open(path).convert("RGB") for path in crop_frame_paths]
        n_before = len(before_images)
        n_target = len(target_images)
        n_crops = len(crop_images)
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"The first {n_before} images are BEFORE CONTEXT and are visually labeled "
                    "BEFORE CONTEXT - DO NOT LABEL.\n"
                    f"The next {n_target} images are full-frame TARGET images and are visually "
                    "labeled TARGET FRAME - LABEL THIS INTERVAL.\n"
                    f"The final {n_crops} images are optional TARGET DETAIL CROPS. They are "
                    "supplemental zoomed-in evidence from TARGET frames only.\n"
                    "Only output actions visible in TARGET frames. Do not output actions visible "
                    "only in BEFORE CONTEXT."
                ),
            }
        ]
        content.extend({"type": "image", "image": image} for image in before_images)
        content.extend({"type": "image", "image": image} for image in target_images)
        if crop_images:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "TARGET DETAIL CROPS. Use these only as supplemental zoomed-in "
                        "evidence consistent with the full-frame TARGET sequence."
                    ),
                }
            )
        content.extend({"type": "image", "image": image} for image in crop_images)
        content.append({"type": "text", "text": prompt})
        return self._generate_from_content(content)
