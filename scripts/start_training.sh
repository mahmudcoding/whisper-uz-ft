#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/usc}"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR"/{data,logs,outputs}

echo "Project: $PROJECT_DIR"
echo "Dataset: $DATASET_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Virtualenv not found; running setup/install.sh first."
  bash "$PROJECT_DIR/setup/install.sh"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

echo "Checking environment..."
bash "$PROJECT_DIR/scripts/check_env.sh" | tee "$LOG_DIR/env_check.log"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: dataset directory does not exist: $DATASET_DIR" >&2
  echo "Place USC data under ~/datasets/usc or set DATASET_DIR=/path/to/usc." >&2
  exit 1
fi

echo "Cleaning and splitting dataset..."
python "$PROJECT_DIR/src/data_loader.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$PROJECT_DIR/data" \
  --report "$LOG_DIR/data_cleaning_report.json" \
  | tee "$LOG_DIR/data_cleaning.log"

for split in train val test; do
  if [[ ! -s "$PROJECT_DIR/data/$split.csv" ]]; then
    echo "ERROR: missing or empty split: $PROJECT_DIR/data/$split.csv" >&2
    exit 1
  fi
done

echo "Running baseline evaluation..."
bash "$PROJECT_DIR/scripts/run_baseline.sh" \
  --data-dir "$PROJECT_DIR/data" \
  --output "$PROJECT_DIR/outputs/baseline_metrics.json" \
  | tee "$LOG_DIR/baseline.log"

echo "Starting partial fine-tuning..."
RESUME_ARG=()
latest_checkpoint="$(find "$PROJECT_DIR/outputs" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
if [[ -n "$latest_checkpoint" ]]; then
  echo "Found checkpoint; resuming from $latest_checkpoint"
  RESUME_ARG=(--resume "$latest_checkpoint")
fi

python "$PROJECT_DIR/src/train.py" --config "$PROJECT_DIR/configs/train.yaml" "${RESUME_ARG[@]}" \
  | tee "$LOG_DIR/training.log"

echo
echo "Training complete."
echo "Final model: $PROJECT_DIR/outputs/final_model"
echo "Baseline metrics: $PROJECT_DIR/outputs/baseline_metrics.json"
echo "Final test metrics: $PROJECT_DIR/outputs/test_metrics.json"
echo "TensorBoard: tensorboard --logdir $PROJECT_DIR/logs/tensorboard"
