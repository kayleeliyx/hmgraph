#!/usr/bin/env bash
# Exactly ten disposable optimizer steps, then a fresh reload. Requires a
# user-confirmed GPU allocation file; never selects a device just because idle.
set -euo pipefail
PROJECT="/data/yaxuanli/ActGround"
RUN="${HDEPIC_SMOKE_RUN:-$PROJECT/runs/hdepic_sft_smoke_8b}"
PYTHON_BIN="$PROJECT/.venvs/hdepic-swift-8b/bin/python"
export PYTHONPATH="$PROJECT" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HF=1
export HF_HOME="$PROJECT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT/.cache/hdepic_swift_dataset"
export MODELSCOPE_CACHE="$PROJECT/.cache/modelscope" SWIFT_CACHE_DIR="$PROJECT/.cache/swift"
export IMAGE_MAX_TOKEN_NUM=144 IMAGE_MIN_TOKEN_NUM=64 OMP_NUM_THREADS=4
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$("$PYTHON_BIN" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a.get("authorized_by_user") is True, "GPU allocation required"; print(a["gpu_uuid"])' "$RUN/gpu_allocation.json")"
"$PYTHON_BIN" -m code.hdepic_preparation.smoke_8b --phase prepare --run "$RUN"
"$PYTHON_BIN" -m code.hdepic_preparation.smoke_8b --phase train --run "$RUN" 2>&1 | tee "$RUN/train.log"
"$PYTHON_BIN" -m code.hdepic_preparation.smoke_8b --phase reload --run "$RUN" 2>&1 | tee "$RUN/reload.log"
