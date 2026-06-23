#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/usc}"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/data" "$PROJECT_DIR/outputs"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  bash "$PROJECT_DIR/setup/install.sh"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

python "$PROJECT_DIR/src/inspect_dataset.py" --dataset-dir "$DATASET_DIR" --report "$LOG_DIR/dataset_structure_report.txt"
python "$PROJECT_DIR/src/verify_gpu.py" --report "$LOG_DIR/gpu_report.txt"
python "$PROJECT_DIR/src/verify_model_freeze.py" --report "$LOG_DIR/model_trainable_report.txt"
python "$PROJECT_DIR/src/validate_audio.py" --dataset-dir "$DATASET_DIR" --report "$LOG_DIR/audio_validation_report.json"
python "$PROJECT_DIR/src/create_mini_splits.py" --dataset-dir "$DATASET_DIR" --out-dir "$PROJECT_DIR/data"

python "$PROJECT_DIR/src/evaluate_baseline.py" \
  --data-dir "$PROJECT_DIR/data" \
  --output "$PROJECT_DIR/outputs/baseline_metrics.json" \
  --splits mini_val mini_test \
  | tee "$LOG_DIR/baseline_mini.log"

python "$PROJECT_DIR/src/train.py" --config "$PROJECT_DIR/configs/dry_run.yaml" \
  | tee "$LOG_DIR/dry_run_training.log"

RESUME_ARG=()
latest_checkpoint="$(find "$PROJECT_DIR/outputs/mini" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "$latest_checkpoint" ]]; then
  RESUME_ARG=(--resume "$latest_checkpoint")
fi

python "$PROJECT_DIR/src/train.py" --config "$PROJECT_DIR/configs/mini_train.yaml" "${RESUME_ARG[@]}" \
  | tee "$LOG_DIR/mini_training.log"
