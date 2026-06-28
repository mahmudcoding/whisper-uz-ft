#!/usr/bin/env bash
set -euo pipefail

cd /home/mahmud/whisper-uz-ft

LOG_DIR="reports/lr_search/blockwise"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/blockwise_phase4_launcher.log"

{
  echo "[$(date -Is)] Waiting for existing LR-search queue before Phase 4 blockwise search."
  while pgrep -af "autonomous_search.py|run_encoder_8e6_addons.sh|run_experiment.py|src/train.py" \
      | grep -v "run_blockwise_phase4.sh" \
      | grep -v "grep" >/dev/null; do
    sleep 60
  done

  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1

  echo "[$(date -Is)] Starting Phase 4 blockwise encoder LR search."
  .venv/bin/python scripts/lr_search/blockwise_search.py
  echo "[$(date -Is)] Phase 4 blockwise encoder LR search complete."
} 2>&1 | tee -a "$LOG_FILE"
