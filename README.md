# Whisper Uzbek ASR

Production-oriented fine-tuning, evaluation, data engineering, inference benchmarking,
and capacity planning for an Uzbek-only `openai/whisper-large-v3` model.

## Objective

Minimize Uzbek WER/CER. Preserving multilingual Whisper capability is not a goal.

Current best completed model:

- registry ID: `partial_ft_usc_baseline`;
- path: `models/partial_ft_usc_baseline/model/`;
- USC test WER: `20.05%`;
- USC test CER: `5.29%`.

The protected baseline must not be modified.

## Current Work

An autonomous decoder/encoder learning-rate and freeze-boundary search is running on
deterministic 10h and 30h Gold-corpus proxies. It uses validation-only selection and
does not load or evaluate test data during search.

Live status:

```bash
tmux has-session -t whisper_lr_search
tail -f reports/lr_search/autonomous_search_console.log
nvidia-smi
```

See [docs/STATUS.md](docs/STATUS.md) for the verified project state.

## Repository Map

| Path | Purpose |
|---|---|
| `src/` | training, model freezing, normalization, data quality, deduplication |
| `scripts/` | launch, monitoring, dataset, transcription, and documentation tools |
| `scripts/lr_search/` | proxy creation, leakage audit, experiment runner/controller |
| `configs/` | training and LR-search YAML configurations |
| `data/` | manifests and derived subsets |
| `benchmark/` | quality evaluation, inference benchmarks, and capacity planning |
| `reports/` | data-quality and LR-search reports |
| `outputs*` | checkpoints and final models |
| `models/` | retained promoted models and their metrics |
| `docs/` | authoritative project documentation |

Raw/staged audio is stored outside Git under `/home/mahmud/datasets/`.

## Setup

Validated host: 52 vCPU, 110 GiB RAM, one NVIDIA A40 48 GB.

```bash
cd /home/mahmud/whisper-uz-ft
bash setup/install.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Validate:

```bash
bash scripts/check_env.sh
.venv/bin/pip check
python -m py_compile src/*.py scripts/*.py scripts/lr_search/*.py benchmark/*.py
python -m text_normalization.tests
python scripts/update_docs.py --check
```

## Core Workflows

Run a training sanity check:

```bash
python src/train.py \
  --config configs/full_ft_uzbek.yaml \
  --sanity-check \
  --sanity-report logs/full_ft_sanity.json
```

Validate LR-search data integrity:

```bash
python scripts/lr_search/validate_lr_subsets.py
python scripts/lr_search/audit_data_leakage.py
```

Run one LR experiment:

```bash
python scripts/lr_search/run_experiment.py \
  --config configs/lr_search/phase1a_decoder_lr_2e6.yaml
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

## Documentation

Start at [docs/README.md](docs/README.md).

Critical guides:

- [Project charter](docs/PROJECT_CHARTER.md)
- [Current status](docs/STATUS.md)
- [Data governance](docs/DATA_GOVERNANCE.md)
- [Training and LR search](docs/TRAINING_AND_SEARCH.md)
- [Operations runbook](docs/OPERATIONS_RUNBOOK.md)
- [Evaluation and benchmarking](docs/EVALUATION_AND_BENCHMARKING.md)
- [Disaster recovery](docs/DISASTER_RECOVERY.md)
- [Agent brief](docs/AGENT_BRIEF.md)

Contributor and Codex instructions are in [AGENTS.md](AGENTS.md).

## Safety

- Never modify `models/partial_ft_usc_baseline/`.
- Never use test data for model selection.
- Never start duplicate controllers against the same output directories.
- Do not train unfiltered Silver/Bronze data.
- Use tmux and checkpoint resume for long jobs.
- Do not store credentials, raw audio, or large checkpoints in Git.
