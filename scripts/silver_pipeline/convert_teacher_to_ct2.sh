#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${SILVER_CONFIG:-$ROOT/configs/silver_datasets.yaml}"

readarray -t TEACHER_CONFIG < <(
  "$ROOT/.venv/bin/python" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["teacher_hf_id"])
print(cfg["teacher_revision"])
print(Path(cfg["teacher_source_model"]).expanduser())
print(Path(cfg["teacher_model"]).expanduser())
PY
)

HF_ID="${TEACHER_CONFIG[0]}"
REVISION="${TEACHER_CONFIG[1]}"
SOURCE="${1:-${TEACHER_CONFIG[2]}}"
DEST="${2:-${TEACHER_CONFIG[3]}}"

if [[ -s "$DEST/model.bin" ]]; then
  echo "Teacher conversion already exists: $DEST"
  exit 0
fi

source "$ROOT/.venv/bin/activate"
if [[ ! -s "$SOURCE/config.json" || ! -s "$SOURCE/model.safetensors" ]]; then
  mkdir -p "$SOURCE"
  hf download "$HF_ID" \
    --repo-type model \
    --revision "$REVISION" \
    --local-dir "$SOURCE" \
    --max-workers 4
fi

if [[ ! -s "$SOURCE/tokenizer.json" ]]; then
  python - "$SOURCE" <<'PY'
import sys
from pathlib import Path

from transformers import WhisperProcessor

source = Path(sys.argv[1])
processor = WhisperProcessor.from_pretrained(source, local_files_only=True)
processor.save_pretrained(source)
if not source.joinpath("tokenizer.json").is_file():
    raise SystemExit(f"Failed to generate {source / 'tokenizer.json'}")
PY
fi

mkdir -p "$DEST"
ct2-transformers-converter \
  --model "$SOURCE" \
  --output_dir "$DEST" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16 \
  --force
printf '%s\n' \
  "Converted SILVER teacher to CTranslate2:" \
  "  model: $HF_ID" \
  "  revision: $REVISION" \
  "  source: $SOURCE" \
  "  destination: $DEST"
