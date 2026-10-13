#!/usr/bin/env bash
# Validation Track generation and scoring for Qwen3.8-27B (base, epoch1, epoch2).
#   generate [models...] --gpus 2,3   one worker on one user-authorized GPU pair
#   score                             score once all three models are complete
# Workers wait for their pair to be idle and never terminate other jobs.
set -euo pipefail
cd /data/yaxuanli/ActGround
source /data/yaxuanli/ActGround/scripts/hdepic_data_storage.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/yaxuanli/ActGround:/data/yaxuanli/ActGround/code
PY=/data/yaxuanli/qwen38-storage/hdepic-qwen38-sft-env/bin/python
RUN=runs/hdepic_qwen38_27b_track_v1
case "${1:-}" in
  generate) shift; exec "$PY" -m code.hdepic_qwen38_track_eval.generate "$@" ;;
  score)
    for model in base epoch1 epoch2; do
      [[ -f $RUN/predictions/$model/completed.json ]] || { echo "Generation incomplete: $model" >&2; exit 2; }
    done
    exec "$PY" evaluation/evaluate.py score --config $RUN/evaluation/config.json ;;
  *) echo "usage: $0 generate [models...] --gpus A,B | score" >&2; exit 2 ;;
esac
