#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
export PYTHONDONTWRITEBYTECODE=1
review_python=/tmp/hdepic-qwen38-27b-env/bin/python
if [[ ! -x "$review_python" ]]; then review_python=/usr/bin/python3; fi
case "${1:-check}" in
  prepare) exec "$review_python" -m code.hdepic_why_semantic_comparison.prepare ;;
  check|report|package) exec "$review_python" -m code.hdepic_why_semantic_comparison.report "$1" ;;
  *) echo 'Usage: run_hdepic_why_semantic_comparison_v2.sh prepare|check|report|package' >&2; exit 2 ;;
esac
