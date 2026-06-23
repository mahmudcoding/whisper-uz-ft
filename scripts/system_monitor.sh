#!/usr/bin/env bash
set -Eeuo pipefail

OUT="${1:-$HOME/whisper-uz-ft/logs/full_training_system.log}"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-5}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
mkdir -p "$(dirname "$OUT")"

echo "timestamp_utc,gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c,gpu_power_w,cpu_pct,ram_used_mb,ram_total_mb,disk_used_pct,disk_avail_gb" >> "$OUT"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gpu=",,,,"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  fi
  cpu="$(python - <<'PY'
import psutil
print(f"{psutil.cpu_percent(interval=0.2):.1f}")
PY
)"
  ram="$(free -m | awk '/^Mem:/ {print $3 "," $2}')"
  disk="$(df -BG "$PROJECT_DIR" | awk 'NR==2 {gsub(/%/,"",$5); gsub(/G/,"",$4); print $5 "," $4}')"
  echo "$ts,$gpu,$cpu,$ram,$disk" >> "$OUT"
  sleep "$INTERVAL"
done
