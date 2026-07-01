# Environment Setup

Last cleaned for obsolete documentation sections: `2026-07-01T04:52:10Z`.

## Observed Environment

- Python 3.12.3
- PyTorch 2.7.1+cu126
- Transformers 5.12.1
- Datasets 5.0.0
- Evaluate 0.4.6
- CUDA 12.6
- GPU: NVIDIA A40 48GB, BF16 supported
- No sudo; use `.venv` and user-space installs.

## Standard Shell

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=/home/mahmud/whisper-uz-ft/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
```

## Validation

```bash
python -m py_compile src/train.py src/model.py
nvidia-smi
df -h /home/mahmud
```
