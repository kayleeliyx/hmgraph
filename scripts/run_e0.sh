#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${ACTGROUND_E0_PYTHON:-/data/yaxuanli/venv/bin/python}"

CONFIG="$PROJECT/configs/e0_pilot.yaml"
if [[ "${1:-}" == "--config" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--config requires a path" >&2
    exit 2
  fi
  CONFIG="$2"
  shift 2
fi

cd "$PROJECT"
exec "$PYTHON_BIN" -m code.e0.run --config "$CONFIG" "$@"
