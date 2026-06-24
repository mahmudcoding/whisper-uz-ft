# Environment and Setup

**Document role:** Recreate and validate the software/hardware environment.

## Validated Host

| Resource | Value |
|---|---|
| OS/kernel | Ubuntu Linux, kernel `6.8.0-101-generic` |
| CPU | 52 vCPU Intel Xeon Ice Lake |
| RAM | 110 GiB |
| Swap | none |
| GPU | NVIDIA A40 48 GB |
| Disk | 2.0 TB root filesystem |
| Privilege | user space; no sudo required |

Observed on 2026-06-24:

- disk used: approximately 840 GB;
- disk available: approximately 1.2 TB;
- CUDA and BF16 available.

## Validated Python Stack

| Package | Version |
|---|---|
| Python | 3.12.3 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| torchaudio | 2.7.1+cu126 |
| CUDA reported by torch | 12.6 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| evaluate | 0.4.6 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| soundfile | 0.14.0 |
| PyYAML | 6.0.3 |

PyTorch must remain at least 2.6 for safe optimizer/scheduler checkpoint loading under
the installed Transformers version. PyTorch 2.5.1 caused resume failure.

## Installation

```bash
cd /home/mahmud/whisper-uz-ft
bash setup/install.sh
```

The installer:

1. selects a usable Python;
2. creates `.venv`;
3. upgrades packaging tools;
4. installs `setup/requirements.lock.txt`;
5. uses the CUDA 12.6 PyTorch package index;
6. attempts optional FlashAttention;
7. runs `scripts/check_env.sh`.

To refresh dependencies rather than use the lock:

```bash
USE_LOCK=0 bash setup/install.sh
```

Do not refresh dependencies during an active training run.

## Runtime Environment

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

For gated Hugging Face assets:

```bash
export HF_TOKEN='...'
```

Never store tokens in the repository, configs, logs, or documentation.

## Verification

```bash
bash scripts/check_env.sh
.venv/bin/pip check
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("torch cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("bf16", torch.cuda.is_bf16_supported())
PY
```

Expected:

- CUDA: `True`;
- GPU: `NVIDIA A40`;
- BF16: `True`;
- no broken package requirements.

## Source Validation

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m py_compile src/*.py scripts/*.py scripts/lr_search/*.py benchmark/*.py
python -m text_normalization.tests
python scripts/update_docs.py --check
git diff --check
```

## Required System Utilities

- `tmux`: persistent training/control sessions.
- `nvidia-smi`: GPU telemetry and process inspection.
- `ffmpeg`: useful for compressed audio; WAV/FLAC can often use SoundFile directly.
- `git`: source and change tracking.

## Environment Failure Modes

### CUDA Not Available

Run `scripts/check_env.sh`, inspect the driver with `nvidia-smi`, and verify the venv is
using CUDA-enabled PyTorch. Do not launch large-v3 training on CPU.

### Resume Fails in `torch.load`

Confirm `torch.__version__ >= 2.6`. Reinstall the pinned lock if needed. Do not
monkeypatch Transformers safety checks.

### Audio Decode Fails

Prefer `soundfile` for local WAV/FLAC. Dataset export scripts use
`datasets.Audio(decode=False)` where TorchCodec is unavailable.

### OOM

Use per-device batch 1, gradient accumulation, BF16, and gradient checkpointing. Check
for another GPU process before changing model configuration.
