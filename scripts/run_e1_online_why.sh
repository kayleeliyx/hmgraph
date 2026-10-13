#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${ACTGROUND_E1_PYTHON:-/data/yaxuanli/venv/bin/python}"
CONFIG="$PROJECT/configs/e1_online_why.yaml"

cd "$PROJECT"
exec "$PYTHON_BIN" -m code.e1_online_why.run --config "$CONFIG" "$@"

