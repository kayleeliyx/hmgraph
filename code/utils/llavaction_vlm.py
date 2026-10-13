"""Single-GPU LLaVAction inference adapter for labeled W frame groups.

The model is loaded once and receives the same ordered BEFORE/TARGET frames
used by the existing W pipeline.  Its compact JSON response is converted by
the caller to the project's W schema.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


PROMPT_ID = "llavaction_w_atomic_v1"

PROMPT = """You are seeing an egocentric cooking video from the camera wearer's view.
The frames are in chronological order. Frames marked BEFORE CONTEXT are only
context. Describe purposeful hand-object actions visible in frames marked
TARGET FRAME only. Do not report an action seen only in BEFORE CONTEXT.

Return ONLY one compact JSON object in this exact form:
{"actions":[{"verb":"short action verb","direct_object":"short object noun","hand":"left|right|both|unknown","confidence":0.0}]}

Use a specific action such as pick up, put down, open, close, press, pour,
stir, mix, grate, cut, write, hold, move, or release. Include at most two
simultaneous actions. If no purposeful hand-object action is visibly occurring
in the TARGET frames, return exactly {"actions":[]}.
"""


class LLaVActionVLM:
    """Load LLaVAction once and generate short structured action records."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        vision_tower_path: str | Path | None = None,
        max_new_tokens: int = 160,
        torch_dtype: str = "bfloat16",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("LLaVAction requires a CUDA GPU for this pipeline")

        from llavaction.model.builder import load_pretrained_model

        self.model_path = str(model_path)
        self.max_new_tokens = int(max_new_tokens)
        overwrite_config = None
        if vision_tower_path is not None:
            overwrite_config = {"mm_vision_tower": str(vision_tower_path)}
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                self.model_path,
                None,
                "llava_qwen",
                torch_dtype=torch_dtype,
                device_map="auto",
                attn_implementation="sdpa",
                overwrite_config=overwrite_config,
            )
        )
        self.model.eval()

    def infer_context(
        self,
        before_frame_paths: Sequence[str],
        target_frame_paths: Sequence[str],
        after_frame_paths: Sequence[str],
        prompt: str = PROMPT,
    ) -> str:
        from llavaction.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llavaction.conversation import conv_templates
        from llavaction.mm_utils import tokenizer_image_token

        all_paths = list(before_frame_paths) + list(target_frame_paths) + list(after_frame_paths)
        if not all_paths:
            raise ValueError("No video frames were supplied to LLaVAction")

        frames = []
        for frame_path in all_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())

        pixel_values = self.image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"]
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        image_tensors = [pixel_values]

        time_instruction = (
            f"The input contains {len(frames)} sampled frames: "
            f"{len(before_frame_paths)} BEFORE CONTEXT, "
            f"{len(target_frame_paths)} TARGET, and "
            f"{len(after_frame_paths)} AFTER CONTEXT frames."
        )
        question = f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n{prompt}"
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        image_sizes = [frame.size for frame in frames]
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        # This model normally returns continuation tokens only.  Keep this
        # guard for transformer versions that return prompt+continuation.
        if generated.shape[1] > input_ids.shape[1] and torch.equal(
            generated[:, : input_ids.shape[1]], input_ids
        ):
            generated = generated[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
