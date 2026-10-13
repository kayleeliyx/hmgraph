#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/data/yaxuanli/venv/bin/python}"

VIDEO="${VIDEO:-/data/kailaic/HDEPIC/data/HD-EPIC/Videos/P01/P01-20240202-110250.mp4}"
VIDEO_BASENAME="$(basename "$VIDEO")"
VIDEO_ID="${VIDEO_ID:-${VIDEO_BASENAME%.*}}"
OUT="${OUT:-$PROJECT/runs/$VIDEO_ID/qwen}"
LOG_DIR="$PROJECT/logs"
LOG="${LOG:-$LOG_DIR/${VIDEO_ID}_qwen_wgroup.log}"
PIDFILE="${PIDFILE:-$LOG_DIR/${VIDEO_ID}_qwen_wgroup.pid}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
START_SEC="${START_SEC:-0}"
END_SEC="${END_SEC:-}"

# This script intentionally starts a fresh Qwen W run. It removes only the
# selected Qwen output directory, never the Gemma/full output directory.
FRESH="${FRESH:-1}"

if [[ "${HMGRAPH_QWEN_INNER:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  export HMGRAPH_QWEN_INNER=1
  export VIDEO VIDEO_ID OUT LOG_DIR LOG PIDFILE PROJECT PYTHON MODEL_ID START_SEC END_SEC FRESH
  nohup bash "$0" > "$LOG" 2>&1 &
  echo $! | tee "$PIDFILE"
  echo "Started fresh Qwen W-group run."
  echo "Video: $VIDEO"
  echo "Output: $OUT"
  echo "Log: $LOG"
  echo "PID: $(cat "$PIDFILE")"
  echo "Monitor: tail -f $LOG"
  exit 0
fi

cd "$PROJECT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -d "$PROJECT/.runtime" ]]; then
  export PYTHONPATH="$PROJECT/.runtime${PYTHONPATH:+:$PYTHONPATH}"
fi
export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "============================================================"
echo "HMGRAPH FRESH QWEN W-GROUP RUN"
echo "============================================================"
date
hostname
echo "PYTHON=$PYTHON"
"$PYTHON" -c 'import sys, transformers; print(f"python_executable={sys.executable}"); print(f"transformers_version={transformers.__version__}")'
echo "VIDEO=$VIDEO"
echo "VIDEO_ID=$VIDEO_ID"
echo "OUT=$OUT"
echo "MODEL_ID=$MODEL_ID"
echo "START_SEC=$START_SEC"
echo "END_SEC=${END_SEC:-full video}"
echo "FRESH=$FRESH"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "HF_HOME=$HF_HOME"
echo

echo "GPU status:"
nvidia-smi || true
echo

if [[ "$FRESH" == "1" ]]; then
  case "$OUT" in
    "$PROJECT"/runs/*/qwen|"$PROJECT"/runs/*/*_qwen)
      echo "Removing existing Qwen output directory: $OUT"
      rm -rf "$OUT"
      ;;
    *)
      echo "Refusing to remove suspicious OUT path: $OUT" >&2
      echo "Set OUT to a path under $PROJECT/runs/... ending in qwen or _qwen." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "$OUT"

w_args=(
  --video-path "$VIDEO"
  --video-id "$VIDEO_ID"
  --output-dir "$OUT"
  --output-jsonl "$OUT/w_outputs.jsonl"
  --model-id "$MODEL_ID"
  --start-sec "$START_SEC"
  --target-fps 8
  --n-frames-per-window 4
  --pre-context-sec 3
  --post-context-sec 0
  --context-fps 2
  --context-mode past
  --overwrite
)

if [[ -n "$END_SEC" ]]; then
  w_args+=(--end-sec "$END_SEC")
fi

echo "Running Qwen W stage..."
"$PYTHON" -m code.w.build "${w_args[@]}"

echo "Building W-group L1 baseline..."
"$PYTHON" build_l1_wgroup.py \
  --w-jsonl "$OUT/w_outputs.jsonl" \
  --out-jsonl "$OUT/l1_candidates_wgroup.jsonl"

echo "============================================================"
echo "DONE"
echo "Output: $OUT"
echo "W output: $OUT/w_outputs.jsonl"
echo "W-group L1: $OUT/l1_candidates_wgroup.jsonl"
echo "============================================================"
date
