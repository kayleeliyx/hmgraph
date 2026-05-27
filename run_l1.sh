#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data2/yaxuanli/hmgraph_codex

VIDEO="${VIDEO:-/data2/yaxuanli/HD-EPIC/Videos/P01/P01-20240202-110250.mp4}"
VIDEO_BASENAME="$(basename "$VIDEO")"
VIDEO_ID="${VIDEO_ID:-${VIDEO_BASENAME%.*}}"
OUT="${OUT:-$PROJECT/runs/$VIDEO_ID/full}"
LOG_DIR="$PROJECT/logs"
LOG="${LOG:-$LOG_DIR/${VIDEO_ID}_l1.log}"
PIDFILE="${PIDFILE:-$LOG_DIR/${VIDEO_ID}_l1.pid}"

# Default: never regenerate W. Set ALLOW_W_REGEN=1 only if you want that.
ALLOW_W_REGEN="${ALLOW_W_REGEN:-0}"

if [[ "${HMGRAPH_INNER:-0}" != "1" ]]; then
  mkdir -p "$OUT" "$LOG_DIR"
  export HMGRAPH_INNER=1
  export VIDEO VIDEO_ID OUT LOG_DIR LOG PIDFILE PROJECT ALLOW_W_REGEN
  nohup bash "$0" > "$LOG" 2>&1 &
  echo $! | tee "$PIDFILE"
  echo "Started hmgraph L1 resume run."
  echo "Video: $VIDEO"
  echo "Output: $OUT"
  echo "Log: $LOG"
  echo "PID: $(cat "$PIDFILE")"
  echo "Monitor: tail -f $LOG"
  exit 0
fi

cd "$PROJECT"

if [[ -f /data/yaxuanli/miniconda3/etc/profile.d/conda.sh ]]; then
  source /data/yaxuanli/miniconda3/etc/profile.d/conda.sh
  conda activate avagemma
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "HMGRAPH L1 RESUME RUN"
echo "============================================================"
date
hostname
which python
echo "VIDEO=$VIDEO"
echo "VIDEO_ID=$VIDEO_ID"
echo "OUT=$OUT"
echo "ALLOW_W_REGEN=$ALLOW_W_REGEN"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo

echo "GPU status:"
nvidia-smi || true
echo

extra_args=()
if [[ "$ALLOW_W_REGEN" == "1" ]]; then
  extra_args+=(--allow-w-regenerate)
fi

python -m code.run_hmgraph_pipeline \
  --video-path "$VIDEO" \
  --output-dir "$OUT" \
  --start-sec 0 \
  --range-label full \
  --max-frames-per-segment 16 \
  --resume-existing \
  --overwrite \
  "${extra_args[@]}"

echo "Running validation..."
python -m code.validate_hmgraph_pipeline_outputs \
  --output-dir "$OUT" \
  --start-sec 0 \
  --video-path "$VIDEO"

echo "============================================================"
echo "DONE"
echo "Output: $OUT"
echo "Final L1: $OUT/l1_candidates.jsonl"
echo "Validation: $OUT/pipeline_validation_report.json"
echo "============================================================"
date
