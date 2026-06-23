#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
VENV_DIR="$PROJECT_DIR/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV_DIR. Run: bash $PROJECT_DIR/setup/install.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
python "$PROJECT_DIR/src/evaluate_baseline.py" "$@"
