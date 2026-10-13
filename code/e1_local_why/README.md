# E1 local why-recognition

This package runs two frozen, fully local conditions on the saved HD-EPIC dev100 split:

- `local_action_clip`: exact saved E0 C1 action-clip frames, presented individually; only the first model decision is used.
- `local_e1`: 16 frames from the fixed five seconds ending at annotated action end, followed by at most three model-selected reads from the available prefix.

The model is the already-downloaded `Qwen/Qwen3-VL-30B-A3B-Instruct` snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, loaded offline in bfloat16 by vLLM 0.23.0 with tensor parallelism across physical GPUs 2 and 3. The host's optional FlashInfer sampling extension is incompatible with its CUDA/CUB headers, so the launcher sets `VLLM_USE_FLASHINFER_SAMPLER=0` and uses vLLM's native deterministic sampler; model attention and MoE kernels remain GPU accelerated. Earlier Transformers attempts loaded the verified weights but produced invalid text and are preserved in separate failed run directories. Every frame is independently resized with aspect ratio preserved and center-padded to a 256×256 RGB canvas. Generation is greedy and deterministic with a 64-token JSON ceiling. No contact sheets, external inference APIs, model judge, or replacement model downloads are used.

The `all` stage loads the model once, runs the accumulated 16/32/48/64-frame capacity check, runs two fixed smoke items without scoring them, resumes/completes both 100-item conditions, audits artifacts, and writes the report.

```bash
cd /data/yaxuanli/ActGround
scripts/run_e1_local_why.sh --stage all
```

Stages can also be run separately with `prepare`, `preflight`, `smoke`, `run`, and `report`. Cached response identity includes the local checkpoint, preprocessing, prompts, generation settings, and exact ordered frame hashes/timestamps.
