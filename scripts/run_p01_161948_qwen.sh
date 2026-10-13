#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/data/yaxuanli/venv/bin/python}"
VIDEO=/data/kailaic/HDEPIC/data/HD-EPIC/Videos/P01/P01-20240202-161948.mp4
VIDEO_ID=P01-20240202-161948
OUT="$PROJECT/runs/$VIDEO_ID/qwen"
MODEL_ID=Qwen/Qwen3-VL-30B-A3B-Instruct
L1_RAW="$OUT/l1_candidates_wgroup.jsonl"
L1_FINAL="$OUT/l1_candidates_wgroup_qwen.jsonl"
OVERLAY="$OUT/overlay_1fps_qwen.mp4"
LOG="$PROJECT/logs/${VIDEO_ID}_qwen_pipeline.log"
PIDFILE="$PROJECT/logs/${VIDEO_ID}_qwen_pipeline.pid"
MIN_FREE_MIB="${MIN_FREE_MIB:-47000}"
REQUIRED_GPUS=2
POLL_SECONDS="${POLL_SECONDS:-45}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"

mkdir -p "$PROJECT/logs"
echo "$$" > "$PIDFILE"

echo "============================================================"
echo "ACTGROUND QWEN W + W-GROUP L1 + VISUALIZATION"
echo "============================================================"
date --iso-8601=seconds
echo "PROJECT=$PROJECT"
echo "VIDEO=$VIDEO"
echo "OUT=$OUT"
echo "MODEL_ID=$MODEL_ID"
echo "PYTHON=$PYTHON"
echo "GPU_CANDIDATES=$GPU_CANDIDATES"
echo "MIN_FREE_MIB=$MIN_FREE_MIB"
echo "REQUIRED_GPUS=$REQUIRED_GPUS"

selected_gpus=()
while (( ${#selected_gpus[@]} < REQUIRED_GPUS )); do
  selected_gpus=()
  for gpu in $GPU_CANDIDATES; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    echo "$(date --iso-8601=seconds) GPU $gpu free=${free_mib}MiB"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      selected_gpus+=("$gpu")
      if (( ${#selected_gpus[@]} == REQUIRED_GPUS )); then
        break
      fi
    fi
  done
  if (( ${#selected_gpus[@]} < REQUIRED_GPUS )); then
    echo "Fewer than $REQUIRED_GPUS GPUs have enough memory; checking again in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
  fi
done

selected_gpu_csv="$(IFS=,; echo "${selected_gpus[*]}")"
echo "Selected physical GPUs $selected_gpu_csv"
export CUDA_VISIBLE_DEVICES="$selected_gpu_csv"
export HMGRAPH_QWEN_INNER=1
export VIDEO VIDEO_ID OUT MODEL_ID PYTHON
export START_SEC=0
export END_SEC=""
export FRESH=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -d "$PROJECT/.runtime" ]]; then
  export PYTHONPATH="$PROJECT/.runtime${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$PROJECT"
bash "$PROJECT/scripts/run_l1_qwen.sh"

if [[ -f "$L1_RAW" ]]; then
  mv "$L1_RAW" "$L1_FINAL"
fi

"$PYTHON" "$PROJECT/scripts/visualize_predictions.py" \
  --video "$VIDEO" \
  --w-jsonl "$OUT/w_outputs.jsonl" \
  --l1-jsonl "$L1_FINAL" \
  --gt-jsonl "$OUT/gt.jsonl" \
  --out "$OVERLAY" \
  --out-fps 1

w_count="$(wc -l < "$OUT/w_outputs.jsonl")"
l1_count="$(wc -l < "$L1_FINAL")"
gt_count="$(wc -l < "$OUT/gt.jsonl")"
overlay_frames="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$OVERLAY")"

echo "W_RECORDS=$w_count"
echo "L1_EVENTS=$l1_count"
echo "GT_EVENTS=$gt_count"
echo "OVERLAY_FRAMES=$overlay_frames"

if [[ "$w_count" -ne 770 ]]; then
  echo "ERROR: expected 770 W records for the 769.6-second video" >&2
  exit 3
fi
if [[ "$gt_count" -ne 297 ]]; then
  echo "ERROR: expected 297 GT events for $VIDEO_ID" >&2
  exit 4
fi
if [[ "$overlay_frames" -ne 770 ]]; then
  echo "ERROR: expected 770 overlay frames" >&2
  exit 5
fi

# Match the preserved Qwen reference-run layout after structured outputs and
# the overlay have passed their acceptance checks.
if [[ -d "$OUT/frames" ]]; then
  rm -rf "$OUT/frames"
fi

echo "============================================================"
echo "PIPELINE COMPLETE"
echo "W=$OUT/w_outputs.jsonl"
echo "L1=$L1_FINAL"
echo "GT=$OUT/gt.jsonl"
echo "CONFIG=$OUT/config.json"
echo "VISUALIZATION=$OVERLAY"
echo "LOG=$LOG"
echo "PIDFILE=$PIDFILE"
echo "============================================================"
date --iso-8601=seconds
