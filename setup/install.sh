#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/whisper-uz-ft}"
VENV_DIR="$PROJECT_DIR/.venv"
LOCK_FILE="$PROJECT_DIR/setup/requirements.lock.txt"

choose_python() {
  for py in python3.11 python3.10 python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      command -v "$py"
      return 0
    fi
  done
  echo "ERROR: no usable Python found in PATH." >&2
  return 1
}

PYTHON_BIN="${PYTHON_BIN:-$(choose_python)}"
echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

mkdir -p "$PROJECT_DIR"/{setup,data,scripts,src,logs,outputs,configs}

create_venv() {
  rm -rf "$VENV_DIR"
  if "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    return 0
  fi

  echo "stdlib venv failed, likely because python3-venv/ensurepip is missing."
  echo "Trying user-space virtualenv bootstrap."
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install --user --upgrade virtualenv --break-system-packages
    "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
    return 0
  fi

  echo "pip is unavailable; installing uv in user space."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "ERROR: need python venv, pip, curl, or wget to bootstrap a user-space Python environment." >&2
    return 1
  fi
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
}

if [[ ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/bin/activate" ]]; then
  create_venv
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -f "$LOCK_FILE" && "${USE_LOCK:-1}" == "1" ]]; then
  echo "Installing pinned dependencies from $LOCK_FILE..."
  pip install -r "$LOCK_FILE" --extra-index-url https://download.pytorch.org/whl/cu121
else
  echo "Installing PyTorch..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
      echo "CUDA PyTorch install failed; falling back to default PyPI wheels."
      pip install --upgrade torch torchvision torchaudio
    fi
  else
    pip install --upgrade torch torchvision torchaudio
  fi

  echo "Installing ASR and MLOps dependencies..."
  pip install --upgrade \
    transformers accelerate datasets evaluate jiwer librosa soundfile numpy pandas \
    sentencepiece tensorboard scikit-learn tqdm peft huggingface_hub pyyaml \
    safetensors

  echo "Installing optional bitsandbytes if available..."
  pip install --upgrade bitsandbytes || echo "bitsandbytes unavailable; training will use standard AdamW."
fi

echo "Installing optional flash-attn if available..."
pip install --upgrade flash-attn --no-build-isolation || echo "flash-attn unavailable; continuing without it."

echo
echo "Environment installed at $VENV_DIR"
"$PROJECT_DIR/scripts/check_env.sh"
