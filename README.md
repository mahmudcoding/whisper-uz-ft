# Whisper Uzbek ASR

Production-grade Uzbek ASR research and engineering built around
`openai/whisper-large-v3`.

## Mission

Build the strongest possible open-weight Uzbek ASR model. The primary optimization
target is Uzbek WER/CER. Preserving multilingual Whisper behavior is not a project
requirement.

## Scope

This repository contains:

- Uzbek text normalization and dataset preparation
- Gold/Silver corpus governance
- Whisper fine-tuning pipelines
- Learning-rate and freezing-strategy experiments
- Evaluation and benchmark tooling
- Inference benchmarking and capacity planning
- Operational documentation for safe long-running training

## Repository Map

| Path | Purpose |
|---|---|
| `src/` | Training, model loading/freezing, normalization, filtering, dedup, scoring |
| `configs/` | Training, LR-search, benchmark, and dataset configuration files |
| `scripts/` | Dataset acquisition/preparation, Silver pipeline, LR search, operations |
| `benchmark/` | Inference benchmarking, evaluation, and capacity planning |
| `data/` | Versioned manifests and derived training/evaluation subsets |
| `reports/` | Generated dataset, search, quality, benchmark, and audit reports |
| `docs/` | Project documentation and decision records |
| `models/` | Protected or promoted model artifacts |
| `outputs*` | Training/search outputs, checkpoints, logs, and metrics |

Large raw/prepared audio is stored outside Git under `/home/mahmud/datasets/`.

## Core Policies

- Test data must not be used for training, LR search, checkpoint selection, or early stopping.
- Uzbek decoding must be forced with `language="uz"` and `task="transcribe"`.
- Canonical Uzbek Latin normalization is the default text target.
- Gold validation/test sets must remain high-trust and protected from leakage.
- Silver data must be filtered and teacher-scored before training.
- Persistent feature caches must be controlled so long runs do not exhaust disk.
- Protected baselines must not be modified or deleted.

## Environment

The project is designed for a CUDA Linux server with an NVIDIA GPU. The current
validated environment uses:

- Python 3.12
- PyTorch with CUDA
- Hugging Face Transformers / Datasets / Evaluate
- NVIDIA A40-class GPU with BF16 support

Typical setup:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
```

Basic validation:

```bash
python -m py_compile src/train.py src/model.py
python -m text_normalization.tests
```

## Documentation

Start with:

- [Status](docs/STATUS.md)
- [Data governance](docs/DATA_GOVERNANCE.md)
- [Training and search](docs/TRAINING_AND_SEARCH.md)
- [Decision log](docs/DECISION_LOG.md)
- [Failure log](docs/FAILURE_LOG.md)
- [Operations runbook](docs/OPERATIONS_RUNBOOK.md)
- [Agent guide](docs/AGENT_BRIEF.md)

Contributor and AI-agent rules are in [AGENTS.md](AGENTS.md).

## Common Commands

Training sanity check:

```bash
python src/train.py --config <config.yaml> --sanity-check --sanity-report logs/sanity.json
```

Run an inference benchmark:

```bash
bash benchmark/scripts/run_benchmark.sh \
  --engine faster-whisper \
  --model-path large-v3 \
  --dataset smoke \
  --precision fp16 \
  --batch-size 1 \
  --beam-size 1 \
  --mode offline
```

Check generated documentation and reports before making experiment decisions.
