#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

python - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "pandas": "pandas",
    "numpy": "numpy",
    "psutil": "psutil",
    "soundfile": "soundfile",
    "jiwer": "jiwer",
    "yaml": "pyyaml",
    "transformers": "transformers",
    "torch": "torch",
    "faster_whisper": "faster-whisper",
    "ctranslate2": "ctranslate2",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *missing])
PY

python benchmark/scripts/create_benchmark_datasets.py

ENGINE="transformers"
MODEL_PATH="openai/whisper-large-v3"
DATASET="smoke"
PRECISION="fp16"
BATCH_SIZE="1"
BEAM_SIZE="1"
MODE="offline"
MAX_SAMPLES=""
FULL_SUITE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --beam-size) BEAM_SIZE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --full-suite) FULL_SUITE="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

run_one() {
  local engine="$1" precision="$2" batch="$3" beam="$4" mode="$5" dataset="$6"
  local extra=()
  if [[ -n "$MAX_SAMPLES" ]]; then
    extra+=(--max-samples "$MAX_SAMPLES")
  fi
  python benchmark/scripts/benchmark_inference.py \
    --engine "$engine" \
    --model-path "$MODEL_PATH" \
    --dataset "$dataset" \
    --precision "$precision" \
    --batch-size "$batch" \
    --beam-size "$beam" \
    --mode "$mode" \
    "${extra[@]}"
}

if [[ "$FULL_SUITE" == "1" ]]; then
  for precision in fp16; do
    for beam in 1 5; do
      for batch in 1 2 4; do
        run_one "$ENGINE" "$precision" "$batch" "$beam" offline "$DATASET"
      done
    done
  done
  for chunk in 1 2 5 10; do
    python benchmark/scripts/benchmark_inference.py \
      --engine "$ENGINE" --model-path "$MODEL_PATH" --dataset "$DATASET" \
      --precision "$PRECISION" --batch-size "$BATCH_SIZE" --beam-size "$BEAM_SIZE" \
      --mode streaming --stream-chunk-seconds "$chunk"
  done
else
  run_one "$ENGINE" "$PRECISION" "$BATCH_SIZE" "$BEAM_SIZE" "$MODE" "$DATASET"
fi

python benchmark/scripts/capacity_planner.py

echo "Benchmark results: $ROOT/benchmark/results"
echo "Capacity report: $ROOT/benchmark/reports/final_capacity_report.md"
