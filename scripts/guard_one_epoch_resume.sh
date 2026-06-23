#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/mahmud/whisper-uz-ft}"
SESSION="${SESSION:-whisper_full_ft_uzbek}"
TRAIN_TARGET="${SESSION}:train"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs_full_ft/checkpoint-1000}"
LOG="${LOG:-${PROJECT_ROOT}/logs/full_ft_one_epoch_guard.log}"
TRAIN_LOG="${TRAIN_LOG:-${PROJECT_ROOT}/logs/full_ft_uzbek.log}"

mkdir -p "$(dirname "$LOG")"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

checkpoint_ready() {
  [ -d "$CHECKPOINT" ] &&
    [ -f "$CHECKPOINT/trainer_state.json" ] &&
    [ -f "$CHECKPOINT/optimizer.pt" ] &&
    [ -f "$CHECKPOINT/scheduler.pt" ]
}

log "guard started; waiting for complete checkpoint: $CHECKPOINT"

while ! checkpoint_ready; do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "ERROR: tmux session $SESSION disappeared before checkpoint was ready"
    exit 1
  fi
  sleep 30
done

log "checkpoint is complete; waiting 60s for filesystem flush"
sleep 60

log "stopping old training pane that was launched with the 4-epoch runtime plan"
tmux send-keys -t "$TRAIN_TARGET" C-c || true

for _ in $(seq 1 24); do
  if ! pgrep -f "python src/train.py --config configs/full_ft_uzbek.yaml --resume auto" >/dev/null; then
    break
  fi
  sleep 5
done

if pgrep -f "python src/train.py --config configs/full_ft_uzbek.yaml --resume auto" >/dev/null; then
  log "training process still alive after SIGINT; sending SIGTERM"
  pkill -TERM -f "python src/train.py --config configs/full_ft_uzbek.yaml --resume auto" || true
  sleep 10
fi

log "restarting training with patched one-epoch config and resume=auto"
tmux respawn-pane -k -t "$TRAIN_TARGET" \
  "cd '$PROJECT_ROOT'; source .venv/bin/activate; export PYTHONPATH=src PYTHONUNBUFFERED=1; echo RESTART_ONE_EPOCH_UTC=\$(date -u +%Y-%m-%dT%H:%M:%SZ) | tee -a '$TRAIN_LOG'; python src/train.py --config configs/full_ft_uzbek.yaml --resume auto 2>&1 | tee -a '$TRAIN_LOG'; echo EXIT_UTC=\$(date -u +%Y-%m-%dT%H:%M:%SZ) | tee -a '$TRAIN_LOG'"

log "restart submitted to tmux target $TRAIN_TARGET"
