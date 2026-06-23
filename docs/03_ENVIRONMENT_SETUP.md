# Environment Setup

## Hardware

Current server:

- CPU: 52 vCPU Intel Xeon Ice Lake.
- RAM: about 110 GiB.
- GPU: 1 x NVIDIA A40 48 GB.
- Disk: 2 TB root volume.
- CUDA: available.
- Sudo: not available/should not be required.

Observed disk state on 2026-06-23:

- Filesystem: `/dev/vda1`, 2.0T total.
- Used: about 701G.
- Available: about 1.3T.

## Python Environment

Project root:

```bash
cd /home/mahmud/whisper-uz-ft
```

Virtual environment:

```bash
source .venv/bin/activate
```

Observed versions:

- Python: `3.12.3`.
- PyTorch: `2.5.1+cu121`.
- CUDA reported by PyTorch: `12.1`.
- Transformers: `5.12.1`.
- datasets: `5.0.0`.
- evaluate: `0.4.6`.
- PyYAML: `6.0.3`.
- soundfile: `0.14.0`.
- NumPy: `2.4.6`.
- GPU: `NVIDIA A40`.
- `torch.cuda.is_bf16_supported()`: `True`.

## Environment Variables

Use these for local scripts:

```bash
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
```

For gated Hugging Face datasets:

```bash
export HF_TOKEN=...
```

or:

```bash
export HUGGINGFACE_HUB_TOKEN=...
```

## Install Dependencies

Use the project venv and install in user space. Do not use sudo.

```bash
cd /home/mahmud/whisper-uz-ft
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers datasets evaluate accelerate soundfile librosa jiwer pyyaml numpy pandas tqdm
python -m pip install faster-whisper ctranslate2 bitsandbytes
```

If a requirements file is later added, prefer:

```bash
python -m pip install -r requirements.txt
```

## System Tools

Expected tools:

- `tmux` for long-running training.
- `nvidia-smi` for GPU telemetry.
- `ffmpeg` is useful but not required by the current dataset export path because audio is decoded with `soundfile` where possible.

## Basic Validation

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src
python -m py_compile src/train.py src/model.py src/text_normalization/uz_normalizer.py
python src/text_normalization/tests.py
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY
```

## Run One Sanity Check

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src
python src/train.py --config configs/full_ft_uzbek.yaml --sanity-check --sanity-report logs/full_ft_sanity_report.json
```
