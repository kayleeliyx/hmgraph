#!/usr/bin/env bash
set -euo pipefail
PROJECT="/data/yaxuanli/ActGround"
DATA="${HDEPIC_DATA_ROOT:-/data/kailaic/HDEPIC/data}"
OUTPUT="${HDEPIC_PREP_OUTPUT:-$PROJECT/datasets/hdepic_sft_v1}"
PYTHON_BIN="${HDEPIC_PREP_PYTHON:-$PROJECT/.venvs/llavaction/bin/python}"
export PYTHONPATH="$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
"$PYTHON_BIN" scripts/build_hdepic_supervision.py --data-root "$DATA" --output "$OUTPUT"
"$PYTHON_BIN" scripts/make_swift_sft.py --output "$OUTPUT" --text-only
"$PYTHON_BIN" -m code.hdepic_preparation.audit_annotations --output "$OUTPUT"
"$PYTHON_BIN" scripts/extract_frames.py --output "$OUTPUT" --sample --workers 4
"$PYTHON_BIN" scripts/make_swift_sft.py --output "$OUTPUT" --sample
"$PYTHON_BIN" scripts/extract_frames.py --output "$OUTPUT" --workers "${HDEPIC_FRAME_WORKERS:-16}" --threads 2
"$PYTHON_BIN" scripts/make_swift_sft.py --output "$OUTPUT"
"$PYTHON_BIN" -m code.hdepic_preparation.report --output "$OUTPUT"
