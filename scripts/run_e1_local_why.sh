#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${ACTGROUND_E1_LOCAL_PYTHON:-/data/kailaic/envs/vllm/bin/python}"
CONFIG="${ACTGROUND_E1_LOCAL_CONFIG:-$PROJECT/configs/e1_local_why.yaml}"

export CUDA_VISIBLE_DEVICES="${ACTGROUND_E1_LOCAL_GPUS:-2,3}"
export PYTHONPATH="$PROJECT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$PROJECT"
exec "$PYTHON_BIN" -m code.e1_local_why.run --config "$CONFIG" "$@"
