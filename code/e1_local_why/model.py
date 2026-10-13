"""Local-only vLLM inference and strict output parsing for E1."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps

from code.e0.dataset import sha256_file
from code.e0.inference import utc_now

from . import LETTERS
from .prompts import SYSTEM_INSTRUCTIONS


def parse_decision(raw_text: str, allow_request: bool) -> dict[str, Any]:
    value = json.loads(raw_text.strip())
    if not isinstance(value, dict) or set(value) != {"answer", "request"}:
        raise ValueError("response must contain exactly answer and request")
    answer = value["answer"]
    if answer not in LETTERS:
        raise ValueError("answer must be one displayed option label")
    request = value["request"]
    if request is None:
        return {"answer": answer, "request": None}
    if not allow_request:
        raise ValueError("response requested video after the read budget was exhausted")
    if not isinstance(request, dict) or set(request) != {"start_sec", "end_sec"}:
        raise ValueError("request must contain exactly start_sec and end_sec")
    start, end = request["start_sec"], request["end_sec"]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (start, end)
    ):
        raise ValueError("requested interval endpoints must be finite numbers")
    return {
        "answer": answer,
        "request": {"start_sec": float(start), "end_sec": float(end)},
    }


class LocalQwenClient:
    """One frozen, tensor-parallel local Qwen3-VL vLLM instance."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        preprocessing: Mapping[str, Any],
        generation: Mapping[str, Any],
    ):
        # Import lazily so parsing/unit tests do not require the vLLM runtime.
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; local E1 requires the assigned GPUs")
        if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
            raise RuntimeError(
                "VLLM_USE_FLASHINFER_SAMPLER=0 is required because this host's "
                "FlashInfer sampler extension is incompatible with its CUDA/CUB headers"
            )
        expected_devices = int(model_config["tensor_parallel_size"])
        if torch.cuda.device_count() != expected_devices:
            raise RuntimeError(
                f"launcher must expose exactly {expected_devices} GPUs; "
                f"found {torch.cuda.device_count()}"
            )
        self._torch = torch
        self._vllm_version = vllm.__version__
        self._SamplingParams = SamplingParams
        self.model_config = dict(model_config)
        self.preprocessing = dict(preprocessing)
        self.generation = dict(generation)
        self.checkpoint = Path(str(model_config["local_path"]))
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"local checkpoint is missing: {self.checkpoint}")
        torch.manual_seed(int(generation["seed"]))
        torch.cuda.manual_seed_all(int(generation["seed"]))
        load_started = time.perf_counter()
        self.model = LLM(
            model=str(self.checkpoint),
            tokenizer=str(self.checkpoint),
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype=str(model_config["precision"]),
            tensor_parallel_size=expected_devices,
            distributed_executor_backend="mp",
            max_model_len=int(model_config["max_model_len"]),
            max_num_seqs=1,
            gpu_memory_utilization=float(model_config["gpu_memory_utilization"]),
            enforce_eager=bool(model_config["enforce_eager"]),
            disable_log_stats=True,
            seed=int(generation["seed"]),
            limit_mm_per_prompt={
                "image": int(model_config["max_images_per_prompt"]),
                "video": 0,
            },
        )
        self.load_latency_seconds = time.perf_counter() - load_started
        self.context_length = int(model_config["max_model_len"])

    def _gpu_memory_snapshot(self) -> list[dict[str, Any]]:
        try:
            import pynvml

            pynvml.nvmlInit()
            result = []
            for visible_index, physical_index in enumerate(
                self.model_config["physical_gpus"]
            ):
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(physical_index))
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result.append(
                    {
                        "visible_index": visible_index,
                        "physical_index": int(physical_index),
                        "used_bytes": int(info.used),
                        "free_bytes": int(info.free),
                        "total_bytes": int(info.total),
                    }
                )
            return result
        except Exception:
            return []

    def environment_details(self) -> dict[str, Any]:
        torch = self._torch
        config_path = self.checkpoint / "config.json"
        preprocessor_path = self.checkpoint / "preprocessor_config.json"
        return {
            "checkpoint_identifier": self.model_config["identifier"],
            "checkpoint_local_path": str(self.checkpoint.resolve()),
            "checkpoint_snapshot": self.checkpoint.name,
            "checkpoint_config_sha256": sha256_file(config_path),
            "preprocessor_config_sha256": sha256_file(preprocessor_path),
            "backend": "vLLM offline tensor parallel",
            "vllm": self._vllm_version,
            "precision": self.model_config["precision"],
            "quantization": {"enabled": False},
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "visible_gpus": [
                {
                    "visible_index": index,
                    "physical_index": self.model_config["physical_gpus"][index],
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
            "tensor_parallel_size": self.model_config["tensor_parallel_size"],
            "gpu_memory_utilization": self.model_config["gpu_memory_utilization"],
            "enforce_eager": self.model_config["enforce_eager"],
            "flashinfer_sampler_enabled": False,
            "model_load_latency_seconds": self.load_latency_seconds,
            "context_length_tokens": self.context_length,
            "local_files_only": True,
            "external_inference_api_used": False,
        }

    def _image(self, path: str) -> Image.Image:
        canvas = int(self.preprocessing["canvas_size"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((canvas, canvas), Image.Resampling.LANCZOS)
            return ImageOps.pad(
                image,
                (canvas, canvas),
                method=Image.Resampling.LANCZOS,
                color=tuple(self.preprocessing["pad_rgb"]),
                centering=(0.5, 0.5),
            )

    def _messages(
        self, plan: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[Image.Image], list[dict[str, Any]]]:
        images: list[Image.Image] = []
        dimensions: list[dict[str, Any]] = []
        content: list[dict[str, Any]] = [
            {"type": "text", "text": str(plan["prompt"])}
        ]
        for batch_index, batch in enumerate(plan["batches"]):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"View {batch_index}, chronological individual source frames:"
                    ),
                }
            )
            for frame_index, frame in enumerate(batch["frames"]):
                image = self._image(str(frame["path"]))
                images.append(image)
                dimensions.append(
                    {
                        "view_index": batch_index,
                        "frame_index": frame_index,
                        "timestamp_sec": float(frame["timestamp_sec"]),
                        "processed_width": image.width,
                        "processed_height": image.height,
                    }
                )
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                f"View {batch_index}, frame {frame_index + 1}, "
                                f"source timestamp "
                                f"{float(frame['timestamp_sec']):.3f} seconds."
                            ),
                        },
                        {"image_pil": image},
                    ]
                )
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": content},
        ]
        return messages, images, dimensions

    def decide(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        requested_at = utc_now()
        started = time.perf_counter()
        messages, images, dimensions = self._messages(plan)
        expected_images = sum(len(batch["frames"]) for batch in plan["batches"])
        if len(images) != expected_images or len(dimensions) != expected_images:
            raise AssertionError("an input frame was dropped before vLLM invocation")
        max_new_tokens = int(self.generation["max_new_tokens"])
        sampling = self._SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            seed=int(self.generation["seed"]),
        )
        generation_started = time.perf_counter()
        try:
            request = self.model.chat(
                messages,
                sampling_params=sampling,
                use_tqdm=False,
                chat_template_content_format="openai",
            )[0]
        finally:
            for image in images:
                image.close()
        generation_latency = time.perf_counter() - generation_started
        if len(request.outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected number of completions")
        completion = request.outputs[0]
        input_token_count = len(request.prompt_token_ids or [])
        generated_tokens = len(completion.token_ids)
        if input_token_count + max_new_tokens > self.context_length:
            raise RuntimeError("context capacity exceeded")
        memory = self._gpu_memory_snapshot()
        peak = max((row["used_bytes"] for row in memory), default=0)
        return {
            "raw_text": completion.text.strip(),
            "request_timestamp": requested_at,
            "latency_seconds": time.perf_counter() - started,
            "processor_latency_seconds": None,
            "generation_latency_seconds": generation_latency,
            "input_token_count": input_token_count,
            "generated_token_count": generated_tokens,
            "context_length_tokens": self.context_length,
            "context_tokens_remaining_after_reserved_output": (
                self.context_length - input_token_count - max_new_tokens
            ),
            "image_count": expected_images,
            "processed_image_dimensions": dimensions,
            "image_grid_thw": None,
            "pixel_values_shape": None,
            "peak_gpu_memory_allocated_bytes": peak,
            "peak_gpu_memory_reserved_bytes": peak,
            "gpu_memory_measurement": (
                "NVML device-used memory observed after the request; GPUs were "
                "idle before model launch"
            ),
            "peak_gpu_memory_by_visible_device": memory,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "cached_prompt_tokens": request.num_cached_tokens,
        }
