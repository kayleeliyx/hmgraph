#!/usr/bin/env bash
# Dataset corrections only. This launcher cannot start model training.
set -euo pipefail
PROJECT="/data/yaxuanli/ActGround"
SOURCE="${HDEPIC_V1_OUTPUT:-$PROJECT/datasets/hdepic_sft_v1}"
PREVIOUS="${HDEPIC_PREVIOUS_V2:-$PROJECT/datasets/hdepic_sft_v2}"
OUTPUT="${HDEPIC_V2_OUTPUT:-$PROJECT/datasets/hdepic_sft_v2_repaired}"
PYTHON_BIN="${HDEPIC_PREP_PYTHON:-$PROJECT/.venvs/llavaction/bin/python}"
SWIFT_PYTHON="${HDEPIC_PROCESSOR_PYTHON:-$PROJECT/.venvs/hdepic-swift-8b/bin/python}"
export PYTHONPATH="$PROJECT" PYTHONDONTWRITEBYTECODE=1
cd "$PROJECT"
"$PYTHON_BIN" -m code.hdepic_preparation.modifier_repair_audit --previous "$PREVIOUS" --output "$OUTPUT" --phase baseline
"$PYTHON_BIN" -m code.hdepic_preparation.preservation --source "$SOURCE" --output "$OUTPUT"
"$PYTHON_BIN" -m unittest code.tests.test_hdepic_preparation code.tests.test_hdepic_revision
"$PYTHON_BIN" -m code.hdepic_preparation.revise --source "$SOURCE" --output "$OUTPUT" --phase build
"$PYTHON_BIN" -m code.hdepic_preparation.revise --output "$OUTPUT" --phase text-audit
"$PYTHON_BIN" -m code.hdepic_preparation.audit_annotations --output "$OUTPUT"
"$PYTHON_BIN" -m code.hdepic_preparation.revise --output "$OUTPUT" --phase sample
"$PYTHON_BIN" -m code.hdepic_preparation.revise --output "$OUTPUT" --phase export
"$SWIFT_PYTHON" -m code.hdepic_preparation.swift_preflight_8b --dataset "$OUTPUT/sample/training20.jsonl" --scoring "$OUTPUT/sample/training20.scoring.jsonl"
"$PYTHON_BIN" -m code.hdepic_preparation.environment --output "$OUTPUT"
"$PYTHON_BIN" -m code.hdepic_preparation.preservation --source "$SOURCE" --output "$OUTPUT" --verify
"$PYTHON_BIN" -m code.hdepic_preparation.modifier_repair_audit --previous "$PREVIOUS" --output "$OUTPUT" --phase report
"$PYTHON_BIN" -m code.hdepic_preparation.package_review --output "$OUTPUT"
