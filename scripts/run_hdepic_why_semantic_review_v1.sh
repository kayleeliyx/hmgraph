#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/data/yaxuanli/ActGround/.venvs/hdepic-swift-8b/bin/python
case "${1:-report}" in
  prepare) "$PYTHON" -m code.hdepic_why_semantic_review.prepare; "$PYTHON" -m code.hdepic_why_semantic_review.pages ;;
  report) "$PYTHON" -m code.hdepic_why_semantic_review.report ;;
  *) echo 'Usage: bash scripts/run_hdepic_why_semantic_review_v1.sh [prepare|report]' >&2; exit 2 ;;
esac
