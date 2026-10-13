#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
source /data/yaxuanli/ActGround/scripts/hdepic_data_storage.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/yaxuanli/ActGround:/data/yaxuanli/ActGround/code
exec /data/yaxuanli/qwen38-storage/hdepic-qwen38-sft-env/bin/python -m code.hdepic_sft_qwen38.workflow "$@"
