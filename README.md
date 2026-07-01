# Whisper Uzbek ASR

Production Uzbek-only ASR fine-tuning, data engineering, evaluation, inference benchmarking, and operations for `openai/whisper-large-v3`.

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

## Mission

Build the best open-weight Uzbek ASR model possible. The only optimization target is Uzbek WER/CER. Preserving multilingual Whisper behavior is not a goal.

## Current Training Direction

The current Stage 1 configuration trains on the Gold+Silver mixed corpus:

```text
configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
```

Configuration summary:

- base model: `openai/whisper-large-v3`
- frozen: encoder layers 0-7
- trainable: encoder layers 8-31 and decoder
- LR: `2e-5` for trainable encoder blocks and decoder
- BF16, batch 4, gradient accumulation 8
- validation/checkpoint every 1000 optimizer steps
- persistent Hugging Face feature caching disabled

See [docs/STATUS.md](docs/STATUS.md) for stable project status and artifact paths.

## Best Preserved Models

- Protected USC partial-FT baseline: `models/partial_ft_usc_baseline/model/`
  - test WER: `20.05%`
  - test CER: `5.29%`
- Best completed Gold-only model: `outputs_full_gold/best_model/`
  - validation WER: `14.50%`
  - validation CER: `3.67%`
  - best step: `5000`

Do not modify or delete the protected baseline.

## Repository Map

| Path | Purpose |
|---|---|
| `src/` | training, model freezing, normalization, filtering, dedup, scoring |
| `configs/` | training, LR-search, and Stage 1 YAML configs |
| `scripts/` | dataset, Silver pipeline, LR-search, and operations scripts |
| `benchmark/` | inference benchmarks and capacity planning |
| `data/` | canonical manifests and derived training subsets |
| `/home/mahmud/datasets/` | staged/prepared audio outside Git |
| `outputs_full_gold/` | completed full-Gold run artifacts |
| `outputs_lr_search/` | LR-search artifacts and metrics |
| `outputs_stage1_gold_silver_nocache/` | Stage 1 Gold+Silver run artifacts |
| `models/` | protected/promoted model artifacts |
| `reports/` | generated dataset/search/benchmark reports |
| `docs/` | authoritative project documentation |

## Setup

Validated environment:

- Python 3.12.3
- PyTorch 2.7.1+cu126
- Transformers 5.12.1
- CUDA 12.6
- NVIDIA A40 48GB, BF16 supported

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=/home/mahmud/whisper-uz-ft/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
```

## Documentation

Start here:

- [Current status](docs/STATUS.md)
- [Data governance](docs/DATA_GOVERNANCE.md)
- [Training and search](docs/TRAINING_AND_SEARCH.md)
- [Decision log](docs/DECISION_LOG.md)
- [Failure log](docs/FAILURE_LOG.md)
- [Operations runbook](docs/OPERATIONS_RUNBOOK.md)
- [Agent brief](docs/AGENT_BRIEF.md)

Contributor and AI-agent operating rules are in [AGENTS.md](AGENTS.md).

## Safety Rules

- Never use test data for training, LR search, early stopping, or checkpoint selection.
- Never modify or delete `models/partial_ft_usc_baseline/`.
- Do not re-enable persistent HF feature caching for large training runs.
- Do not use this project's fine-tuned model as the Silver teacher.
