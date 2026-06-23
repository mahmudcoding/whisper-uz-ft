#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
LOG_FILE="$PROJECT_DIR/logs/system_monitor.log"
INTERVAL="${INTERVAL:-10}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Logging system monitor to $LOG_FILE every ${INTERVAL}s"
while true; do
  {
    echo "===== $(date -Is) ====="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits || true
    else
      echo "nvidia-smi not found"
    fi
    echo "-- CPU/RAM --"
    top -bn1 | sed -n '1,5p' || true
    free -h || true
    echo "-- DISK --"
    df -h "$PROJECT_DIR" "$HOME/datasets/usc" 2>/dev/null || df -h "$PROJECT_DIR"
    echo
  } | tee -a "$LOG_FILE"
  sleep "$INTERVAL"
done
