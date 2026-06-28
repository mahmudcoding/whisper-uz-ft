#!/usr/bin/env bash
set -euo pipefail

cd /home/mahmud/whisper-uz-ft

FOLLOWUP_PID="${FOLLOWUP_PID:-2521483}"
CURRENT_ID="${CURRENT_ID:-phase4x_main_decoder_1p6em05}"
TARGET_ID="${TARGET_ID:-phase4x_main_all_blocks_aggressive}"
CURRENT_METRICS="outputs_lr_search/${CURRENT_ID}/metrics.json"
LOG_DIR="reports/lr_search"
LOG_FILE="$LOG_DIR/phase4_all_blocks_aggressive_followup.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

resume_followup() {
  if kill -0 "$FOLLOWUP_PID" 2>/dev/null; then
    kill -CONT "$FOLLOWUP_PID" || true
  fi
}
trap resume_followup EXIT INT TERM

if ! kill -0 "$FOLLOWUP_PID" 2>/dev/null; then
  echo "[$(date -Is)] Follow-up PID $FOLLOWUP_PID is not running; refusing to reorder queue." >&2
  exit 1
fi

echo "[$(date -Is)] Waiting for current run metrics: $CURRENT_METRICS"
until [[ -f "$CURRENT_METRICS" ]]; do
  sleep 5
done

kill -STOP "$FOLLOWUP_PID"
echo "[$(date -Is)] Paused follow-up controller $FOLLOWUP_PID after current metrics were written."

while pgrep -af "run_experiment.py.*${CURRENT_ID}" >/dev/null; do
  sleep 5
done

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -f "outputs_lr_search/${TARGET_ID}/metrics.json" ]]; then
  echo "[$(date -Is)] Target metrics already exist for ${TARGET_ID}; skipping duplicate launch."
else
  echo "[$(date -Is)] Starting requested all-layer blockwise 30h run: ${TARGET_ID}"
  .venv/bin/python scripts/lr_search/run_experiment.py \
    --config configs/lr_search/blockwise/all_blocks_aggressive_main.yaml \
    --experiment-id "$TARGET_ID"
fi

echo "[$(date -Is)] Requested all-layer blockwise run complete; resuming previous queue."
resume_followup
trap - EXIT INT TERM
