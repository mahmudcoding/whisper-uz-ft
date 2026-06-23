# Whisper Uzbek Fine-Tuning

This project fine-tunes `openai/whisper-large-v3` for Uzbek Latin-script ASR in user space, with no sudo required.

## Layout

- `setup/install.sh`: creates `.venv` and installs Python, ASR, and MLOps dependencies.
- `scripts/check_env.sh`: prints OS, Python, GPU, CUDA, build tools, disk, RAM, and Torch CUDA status.
- `src/data_loader.py`: ingests USC data, normalizes transcripts, removes bad samples, and creates splits.
- `src/train.py`: partial fine-tuning pipeline with WER/CER, checkpointing, early stopping, TensorBoard logs, and best-checkpoint restore.
- `scripts/start_training.sh`: one-command setup check, data cleaning, baseline evaluation, and training.
- `scripts/transcribe.py`: inference from a trained checkpoint.
- `scripts/monitor.sh`: GPU/CPU/RAM/disk monitor logging to `logs/system_monitor.log`.

## Dataset

Place Uzbek Speech Corpus files under:

```bash
~/datasets/usc/
```

Supported layouts:

- CSV/JSON/JSONL metadata with audio columns like `audio`, `audio_path`, `path`, `file`, `filename`.
- Transcript columns named `text`, `transcript`, `sentence`, or `transcription`.
- Optional speaker columns like `speaker_id`, `speaker`, `client_id`.
- Audio folders with `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, or `.aac` plus same-name `.txt` or `.lab` sidecars.

## Setup

```bash
cd ~/whisper-uz-ft
bash setup/install.sh
```

The installer chooses `python3.11`, then `python3.10`, then system `python3/python`, creates `~/whisper-uz-ft/.venv`, installs CUDA PyTorch when possible, and skips optional packages that cannot install.
By default it installs pinned versions from `setup/requirements.lock.txt`; set `USE_LOCK=0` to refresh to current package releases.

## Train

```bash
cd ~/whisper-uz-ft
bash scripts/start_training.sh
```

The launcher runs environment checks, cleans and splits the dataset, evaluates raw Whisper large-v3 on validation/test, then starts partial fine-tuning.

Defaults are in `configs/train.yaml`:

- Train full decoder and last 8 encoder blocks.
- Freeze early encoder blocks.
- FP16, gradient checkpointing, gradient accumulation.
- AdamW, cosine scheduler, WER/CER evaluation.
- Best-checkpoint restore and early stopping after 5 evaluations.

## Resume

`scripts/start_training.sh` automatically resumes from the latest `outputs/checkpoint-*` directory. To resume manually:

```bash
source ~/whisper-uz-ft/.venv/bin/activate
export PYTHONPATH=~/whisper-uz-ft/src
python ~/whisper-uz-ft/src/train.py --config ~/whisper-uz-ft/configs/train.yaml --resume ~/whisper-uz-ft/outputs/checkpoint-500
```

## Monitor

Run in another terminal:

```bash
cd ~/whisper-uz-ft
bash scripts/monitor.sh
```

## Inference

```bash
source ~/whisper-uz-ft/.venv/bin/activate
python ~/whisper-uz-ft/scripts/transcribe.py /path/to/audio.wav --checkpoint ~/whisper-uz-ft/outputs/final_model --beam-size 5
```

## Troubleshooting

- Missing CUDA: run `scripts/check_env.sh`. Training requires CUDA by default. Set `require_cuda: false` only for small CPU smoke tests.
- OOM: set `per_device_batch_size: 1` in `configs/train.yaml`; keep or increase `gradient_accumulation_steps`.
- Missing `ffmpeg`: WAV/FLAC often work through `soundfile`; MP3/M4A may require a user-space ffmpeg binary or conda/micromamba environment.
- Empty splits: inspect `logs/data_cleaning_report.json`; common causes are wrong metadata column names, missing audio paths, or durations outside 1-30 seconds.
- Bad baseline WER: inspect several rows in `data/train.csv`, `data/val.csv`, and `data/test.csv` before training.
