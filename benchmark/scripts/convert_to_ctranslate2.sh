#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
MODEL_PATH="${1:-openai/whisper-large-v3}"
OUT_DIR="${2:-$PROJECT_DIR/benchmark/models/$(basename "$MODEL_PATH")-ct2}"
QUANTIZATION="${QUANTIZATION:-float16}"

source "$PROJECT_DIR/.venv/bin/activate"
mkdir -p "$OUT_DIR"

ct2-transformers-converter \
  --model "$MODEL_PATH" \
  --output_dir "$OUT_DIR" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization "$QUANTIZATION"

echo "$OUT_DIR"
