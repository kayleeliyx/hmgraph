#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data2/yaxuanli/hmgraph_codex
VIDEO_ID=P01-20240202-110250
VISIBLE_GPUS="${VISIBLE_GPUS:-0,3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/runs/$VIDEO_ID/qa_ravu_qwen3_fixed_v1}"
LOG_DIR="$PROJECT/logs"
LOG="${LOG:-$LOG_DIR/${VIDEO_ID}_ravu_qwen3_fixed_v1.log}"
PIDFILE="${PIDFILE:-$LOG_DIR/${VIDEO_ID}_ravu_qwen3_fixed_v1.pid}"
PYTHON=/data/yaxuanli/miniconda3/envs/avagemma/bin/python

if [[ "${HMGRAPH_RAVU_INNER:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
  export HMGRAPH_RAVU_INNER=1 PROJECT VIDEO_ID VISIBLE_GPUS OUTPUT_DIR LOG_DIR LOG PIDFILE PYTHON
  nohup bash "$0" >"$LOG" 2>&1 &
  pid=$!
  echo "$pid" | tee "$PIDFILE"
  echo "Started resumable RAVU-style W/L1/W+L1 evaluation."
  echo "Physical GPUs: $VISIBLE_GPUS"
  echo "Output: $OUTPUT_DIR"
  echo "Log: $LOG"
  echo "PID: $pid"
  exit 0
fi

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "HMGRAPH RAVU-STYLE FIXED QA"
echo "============================================================"
date
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OUTPUT_DIR=$OUTPUT_DIR"
nvidia-smi || true

"$PYTHON" -m code.evaluate_ravu_graph_qa \
  --config configs/ravu_qwen3_fixed_v1.json \
  --annotation-root /data2/yaxuanli/HD-EPIC/hd-epic-annotations/vqa-benchmark \
  --w-jsonl runs/P01-20240202-110250/qwen/w_outputs.jsonl \
  --l1-jsonl runs/P01-20240202-110250/qwen/l1_candidates_wgroup_qwen.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --modes w l1 w+l1

echo "============================================================"
echo "DONE"
echo "Output: $OUTPUT_DIR"
date
