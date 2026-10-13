"""Minimal local vLLM tensor-parallel checkpoint sanity check."""

from __future__ import annotations

from vllm import LLM, SamplingParams


CHECKPOINT = (
    "/data/yaxuanli/ActGround/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)


def main() -> None:
    model = LLM(
        model=CHECKPOINT,
        tokenizer=CHECKPOINT,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=False,
        limit_mm_per_prompt={"image": 64, "video": 0},
    )
    result = model.chat(
        [{"role": "user", "content": "Who are you?"}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        use_tqdm=False,
    )
    print("finish_reason", result[0].outputs[0].finish_reason)
    print("decoded", repr(result[0].outputs[0].text))


if __name__ == "__main__":
    main()
