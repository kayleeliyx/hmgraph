#!/usr/bin/env bash
# Step-hint experiment for Qwen3.8-27B (base, epoch1) on 650 validation Track queries.
#   prepare                      CPU: build inputs (selection, references, prompts)
#   generate MODEL --gpus A,B    GPU: one worker per model on one user-authorized GPU pair
#   analyze                      CPU: score with the standalone evaluator and write report.md
set -euo pipefail
cd /data/yaxuanli/ActGround
source /data/yaxuanli/ActGround/scripts/hdepic_data_storage.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/yaxuanli/ActGround:/data/yaxuanli/ActGround/code
PY=/data/yaxuanli/qwen38-storage/hdepic-qwen38-sft-env/bin/python
case "${1:-}" in
  prepare) exec "$PY" -m code.hdepic_qwen38_step_hint.prepare ;;
  generate) shift; exec "$PY" -m code.hdepic_qwen38_step_hint.run "$@" ;;
  analyze) exec "$PY" -m code.hdepic_qwen38_step_hint.analyze ;;
  *) echo "usage: $0 prepare | generate MODEL --gpus A,B | analyze" >&2; exit 2 ;;
esac
