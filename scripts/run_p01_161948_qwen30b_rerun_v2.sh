#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/data/yaxuanli/venv/bin/python}"
VIDEO=/data/kailaic/HDEPIC/data/HD-EPIC/Videos/P01/P01-20240202-161948.mp4
VIDEO_ID=P01-20240202-161948
MODEL_ID=Qwen/Qwen3-VL-30B-A3B-Instruct
OUT="$PROJECT/runs/$VIDEO_ID/qwen30b_rerun_v2"
RAW="$OUT/raw_inference"
PROCESSED="$OUT/processed"
REFERENCE="$PROJECT/runs/$VIDEO_ID/qwen"
LOG="$PROJECT/logs/${VIDEO_ID}_qwen30b_rerun_v2.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-47000}"

if [[ -e "$OUT" ]]; then
  echo "ERROR: isolated rerun directory already exists: $OUT" >&2
  exit 2
fi
if [[ ! -s "$VIDEO" ]]; then
  echo "ERROR: input video missing or empty: $VIDEO" >&2
  exit 3
fi
for gpu in 0 1; do
  free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  echo "GPU $gpu free=${free_mib}MiB"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
    echo "ERROR: GPU $gpu needs at least ${MIN_FREE_MIB}MiB free" >&2
    exit 4
  fi
done

mkdir -p "$PROJECT/logs" "$RAW"
exec > >(tee -a "$LOG") 2>&1

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -d "$PROJECT/.runtime" ]]; then
  export PYTHONPATH="$PROJECT/.runtime${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$PROJECT"
echo "ACTGROUND QWEN-30B CLEAN RERUN V2"
date --iso-8601=seconds
echo "VIDEO=$VIDEO"
echo "MODEL_ID=$MODEL_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OUT=$OUT"

"$PYTHON" -m code.w.build \
  --video-path "$VIDEO" \
  --video-id "$VIDEO_ID" \
  --output-dir "$RAW" \
  --output-jsonl "$RAW/w_outputs.jsonl" \
  --model-id "$MODEL_ID" \
  --start-sec 0 \
  --target-fps 8 \
  --n-frames-per-window 4 \
  --pre-context-sec 3 \
  --post-context-sec 0 \
  --context-fps 2 \
  --context-mode past \
  --device auto \
  --overwrite

"$PYTHON" build_l1_wgroup.py \
  --w-jsonl "$RAW/w_outputs.jsonl" \
  --out-jsonl "$PROCESSED/l1_candidates_wgroup_qwen.jsonl" \
  --baseline-l1 "$REFERENCE/l1_candidates_wgroup_qwen.jsonl"

cp "$PROCESSED/w_reprocessed.jsonl" "$OUT/w_outputs.jsonl"
cp "$PROCESSED/l1_candidates_wgroup_qwen.jsonl" "$OUT/l1_candidates_wgroup_qwen.jsonl"
cp "$RAW/gt.jsonl" "$OUT/gt.jsonl"
cp "$RAW/config.json" "$OUT/config.json"
cp "$RAW/alias_table.json" "$OUT/alias_table.json"

"$PYTHON" "$PROJECT/scripts/compare_qwen_runs.py" \
  --baseline-w "$REFERENCE/w_outputs.jsonl" \
  --baseline-l1 "$REFERENCE/l1_candidates_wgroup_qwen.jsonl" \
  --candidate-w "$OUT/w_outputs.jsonl" \
  --candidate-l1 "$OUT/l1_candidates_wgroup_qwen.jsonl" \
  --gt "$OUT/gt.jsonl" \
  --alias-table "$OUT/alias_table.json" \
  --baseline-name previous_qwen30b \
  --candidate-name rerun_qwen30b_reprocessed_v2 \
  --out-json "$OUT/comparison_to_previous_qwen.json" \
  --out-csv "$OUT/comparison_per_gt_event.csv"

"$PYTHON" "$PROJECT/scripts/visualize_predictions.py" \
  --video "$VIDEO" \
  --w-jsonl "$OUT/w_outputs.jsonl" \
  --l1-jsonl "$OUT/l1_candidates_wgroup_qwen.jsonl" \
  --gt-jsonl "$OUT/gt.jsonl" \
  --out "$OUT/overlay_1fps_qwen.mp4" \
  --out-fps 1

w_count="$(wc -l < "$OUT/w_outputs.jsonl")"
raw_w_count="$(wc -l < "$RAW/w_outputs.jsonl")"
l1_count="$(wc -l < "$OUT/l1_candidates_wgroup_qwen.jsonl")"
gt_count="$(wc -l < "$OUT/gt.jsonl")"
overlay_frames="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$OUT/overlay_1fps_qwen.mp4")"
echo "RAW_W_RECORDS=$raw_w_count"
echo "REPROCESSED_W_RECORDS=$w_count"
echo "L1_EVENTS=$l1_count"
echo "GT_EVENTS=$gt_count"
echo "OVERLAY_FRAMES=$overlay_frames"
[[ "$raw_w_count" -eq 770 && "$w_count" -eq 770 ]] || exit 5
[[ "$gt_count" -eq 297 ]] || exit 6
[[ "$overlay_frames" -eq 770 ]] || exit 7

if [[ -d "$RAW/frames" ]]; then
  rm -rf "$RAW/frames"
fi

echo "PIPELINE COMPLETE"
echo "W=$OUT/w_outputs.jsonl"
echo "RAW_W=$RAW/w_outputs.jsonl"
echo "L1=$OUT/l1_candidates_wgroup_qwen.jsonl"
echo "VISUALIZATION=$OUT/overlay_1fps_qwen.mp4"
echo "COMPARISON=$OUT/comparison_to_previous_qwen.json"
echo "LOG=$LOG"
date --iso-8601=seconds
