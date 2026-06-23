#!/usr/bin/env bash
set -Eeuo pipefail

echo "== OS =="
if [[ -f /etc/os-release ]]; then
  cat /etc/os-release
else
  uname -a
fi

echo
echo "== Kernel =="
uname -a

echo
echo "== Python =="
for py in python3.11 python3.10 python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    printf "%-12s %s -> %s\n" "$py" "$(command -v "$py")" "$($py --version 2>&1)"
  else
    printf "%-12s not found\n" "$py"
  fi
done

echo
echo "== Build tools =="
for bin in gcc g++ make cmake ffmpeg git; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "$bin: $(command -v "$bin")"
    "$bin" --version 2>&1 | head -n 1 || true
  else
    echo "$bin: NOT FOUND"
  fi
done

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi: NOT FOUND"
fi

echo
echo "== Disk =="
df -h "$HOME" .

echo
echo "== RAM =="
free -h || true

echo
echo "== Torch CUDA =="
PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PY="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY="$(command -v python)"
fi

"$PY" - <<'PY' || true
import sys
print("python:", sys.version.replace("\n", " "))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("cuda devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0))
        x = torch.ones((1024, 1024), device="cuda")
        print("cuda tensor test:", float(x.sum().item()))
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
