# Environment Setup

Last rebuilt from repository reality: `2026-07-01T04:52:10Z`.

For current live state, read `STATUS.md` first. For full memory transfer, read `../PROJECT_CONTEXT_EXPORT.txt`.

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
tmux ls
nvidia-smi
df -h /home/mahmud
```
