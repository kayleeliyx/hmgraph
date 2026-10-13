#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT/.venvs/llavaction/bin/python"
VIDEO=/data/kailaic/HDEPIC/data/HD-EPIC/Videos/P01/P01-20240202-161948.mp4
VIDEO_ID=P01-20240202-161948
ANNOTATIONS=/data/kailaic/HDEPIC/data/hd-epic-annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl
VOCAB=/data/kailaic/HDEPIC/data/hd-epic-annotations/narrations-and-action-segments
MODEL="$PROJECT/.cache/huggingface/models--MLAdaptiveIntelligence--LLaVAction-7B/snapshots/4b43e5223635402c98e5ba51b07ca1b7a7b96b0a"
VISION_TOWER=google/siglip-so400m-patch14-384
OUT="$PROJECT/runs/$VIDEO_ID/llavaction/pilot_200_290"
QWEN_W="$PROJECT/runs/$VIDEO_ID/qwen/w_outputs.jsonl"
GPU_ID="${GPU_ID:-0}"
START_SEC=200
END_SEC=290
L1="$OUT/l1_candidates_wgroup_llavaction.jsonl"
OVERLAY="$OUT/overlay_1fps_llavaction.mp4"
LOG="$OUT/pipeline.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "ACTGROUND LLAVACTION-7B PILOT"
echo "============================================================"
date --iso-8601=seconds
echo "GPU=$GPU_ID VIDEO=$VIDEO RANGE=$START_SEC-$END_SEC OUT=$OUT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

cd "$PROJECT"
"$PYTHON" -u -m code.w.build_llavaction \
  --video-path "$VIDEO" \
  --video-id "$VIDEO_ID" \
  --output-dir "$OUT" \
  --model-path "$MODEL" \
  --vision-tower-path "$VISION_TOWER" \
  --annotations-path "$ANNOTATIONS" \
  --hdepic-vocab-path "$VOCAB" \
  --start-sec "$START_SEC" \
  --end-sec "$END_SEC"

"$PYTHON" "$PROJECT/build_l1_wgroup.py" \
  --w-jsonl "$OUT/w_outputs.jsonl" \
  --out-jsonl "$L1" \
  --allow-no-bimanual

"$PYTHON" "$PROJECT/scripts/visualize_predictions.py" \
  --video "$VIDEO" \
  --w-jsonl "$OUT/w_outputs.jsonl" \
  --l1-jsonl "$L1" \
  --gt-jsonl "$OUT/gt.jsonl" \
  --out "$OVERLAY" \
  --out-fps 1 \
  --start-sec "$START_SEC" \
  --end-sec "$END_SEC"

"$PYTHON" "$PROJECT/scripts/evaluate_w_pilot.py" \
  --candidate "$OUT/w_outputs.jsonl" \
  --baseline "$QWEN_W" \
  --gt "$OUT/gt.jsonl" \
  --alias-table "$OUT/alias_table.json" \
  --start-sec "$START_SEC" \
  --end-sec "$END_SEC" \
  --out-json "$OUT/pilot_comparison.json" \
  --out-csv "$OUT/pilot_comparison_windows.csv"

"$PYTHON" -m pip freeze > "$OUT/environment.txt"

w_count="$(wc -l < "$OUT/w_outputs.jsonl")"
l1_count="$(wc -l < "$L1")"
gt_count="$(wc -l < "$OUT/gt.jsonl")"
overlay_frames="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$OVERLAY")"
echo "W_RECORDS=$w_count L1_EVENTS=$l1_count GT_EVENTS=$gt_count OVERLAY_FRAMES=$overlay_frames"
[[ "$w_count" -eq 90 ]]
[[ "$gt_count" -eq 297 ]]
[[ "$overlay_frames" -eq 90 ]]

echo "PIPELINE COMPLETE: $OUT"
date --iso-8601=seconds
