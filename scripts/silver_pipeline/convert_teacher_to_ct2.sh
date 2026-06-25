#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${1:-$ROOT/archive/partial_ft_usc/model/final_model}"
DEST="${2:-$ROOT/archive/partial_ft_usc/model_ct2}"

if [[ -s "$DEST/model.bin" ]]; then
  echo "Teacher conversion already exists: $DEST"
  exit 0
fi

source "$ROOT/.venv/bin/activate"
mkdir -p "$DEST"
ct2-transformers-converter \
  --model "$SOURCE" \
  --output_dir "$DEST" \
  --copy_files tokenizer.json preprocessor_config.json processor_config.json \
  --quantization float16 \
  --force
echo "Converted independent USC-only teacher to CTranslate2: $DEST"
