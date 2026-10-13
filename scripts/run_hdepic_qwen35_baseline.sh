#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
export PYTHONDONTWRITEBYTECODE=1
TASK_PY=/data/yaxuanli/ActGround/.venvs/hdepic-qwen35-35b/bin/python
TASK_RUN=/data/yaxuanli/ActGround/runs/hdepic_qwen35_35b_baseline_v1
TASK_STAGE=${1:-all}
mkdir -p "$TASK_RUN"
# A second invocation cannot compete with this baseline for either GPU.
exec 9>"$TASK_RUN/pipeline.lock"
flock -n 9 || { echo 'This baseline launcher is already active.' >&2; exit 2; }
test -x "$TASK_PY" || { echo 'Run scripts/prepare_hdepic_qwen35_environment.sh first.' >&2; exit 2; }
prepare() {
  "$TASK_PY" -m code.hdepic_qwen35_baseline.data
  "$TASK_PY" -m code.hdepic_qwen35_baseline.model
  "$TASK_PY" -m code.hdepic_qwen35_baseline.processor --model "$TASK_RUN/model_metadata" --run "$TASK_RUN"
  "$TASK_PY" -m code.hdepic_qwen35_baseline.runner freeze --run "$TASK_RUN"
}
infer() {
  if "$TASK_PY" -m code.hdepic_qwen35_baseline.runner infer --run "$TASK_RUN"; then
    return 0
  else
    "$TASK_PY" -m code.hdepic_qwen35_baseline.report --run "$TASK_RUN"
    return 2
  fi
}
case "$TASK_STAGE" in
  prepare) prepare ;;
  infer) infer ;;
  integrity) "$TASK_PY" -m code.hdepic_qwen35_baseline.weight_integrity --run "$TASK_RUN" ;;
  review) "$TASK_PY" -m code.hdepic_qwen35_baseline.review --run "$TASK_RUN" ;;
  report) "$TASK_PY" -m code.hdepic_qwen35_baseline.report --run "$TASK_RUN" ;;
  package) "$TASK_PY" -m code.hdepic_qwen35_baseline.package --run "$TASK_RUN" ;;
  all)
    prepare
    infer
    "$TASK_PY" -m code.hdepic_qwen35_baseline.weight_integrity --run "$TASK_RUN"
    "$TASK_PY" -m code.hdepic_qwen35_baseline.review --run "$TASK_RUN"
    "$TASK_PY" -m code.hdepic_qwen35_baseline.report --run "$TASK_RUN"
    "$TASK_PY" -m code.hdepic_qwen35_baseline.package --run "$TASK_RUN"
    ;;
  *) echo 'Usage: run_hdepic_qwen35_baseline.sh {prepare|infer|integrity|review|report|package|all}' >&2; exit 2 ;;
esac
