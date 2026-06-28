#!/usr/bin/env bash
set -euo pipefail

cd /home/mahmud/whisper-uz-ft

LOG_DIR="reports/lr_search"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/encoder_8e6_addons.log"

{
  echo "[$(date -Is)] Waiting for active LR-search training to finish before 8e-6 add-on runs."
  while pgrep -af "scripts/lr_search/autonomous_search.py|scripts/lr_search/run_experiment.py|src/train.py" \
      | grep -v "run_encoder_8e6_addons.sh" \
      | grep -v "grep" >/dev/null; do
    sleep 60
  done

  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1

  echo "[$(date -Is)] Running upper encoder 24-31 add-on with encoder_lr=8e-6."
  .venv/bin/python scripts/lr_search/run_experiment.py \
    --config configs/lr_search/upper_encoder_lr_8e6.yaml \
    --experiment-id phase2_upper_encoder_8em06

  echo "[$(date -Is)] Regenerating LR-search comparison."
  .venv/bin/python scripts/lr_search/compare_experiments.py
  echo "[$(date -Is)] 8e-6 add-on runs complete."
} 2>&1 | tee -a "$LOG_FILE"
