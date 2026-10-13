#!/usr/bin/env bash
set -euo pipefail
cd /data/yaxuanli/ActGround
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/yaxuanli/ActGround
exec /data/yaxuanli/ActGround/.venvs/hdepic-swift-8b/bin/python -m code.hdepic_experiment.workflow "$@"
