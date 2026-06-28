# Repository Guidelines

## Mission and Scope

This repository builds a dedicated Uzbek ASR system from
`openai/whisper-large-v3`. The only optimization target is Uzbek WER/CER.
Catastrophic forgetting of non-Uzbek languages is acceptable.

## Required Startup Checklist

Before changing code, configs, data, docs, or running jobs:

```bash
cd /home/mahmud/whisper-uz-ft
git status --short
tmux ls
pgrep -af 'src/train.py|run_experiment.py|score_teacher.py|autonomous_search.py'
nvidia-smi
df -h .
```

Then read:

1. `docs/AGENT_BRIEF.md`
2. `docs/STATUS.md`
3. `docs/DECISION_LOG.md`
4. `docs/FAILURE_LOG.md`
5. `docs/DATA_GOVERNANCE.md`
6. `docs/TRAINING_AND_SEARCH.md`

Live artifacts are authoritative when docs disagree. Update docs after reconciling.

## Non-Negotiable Rules

- Do not modify or delete `models/partial_ft_usc_baseline/`.
- Do not use test data for training, LR search, early stopping, checkpoint selection,
  or model ranking.
- Always force `language="uz"` and `task="transcribe"`.
- Do not train unfiltered Silver or Bronze data.
- Do not use this project’s fine-tuned model as the Silver teacher; use
  `Kotib/uzbek_stt_v1` or another independent validated teacher.
- Do not launch duplicate long-running training jobs.
- Do not run `git reset --hard` or revert unrelated dirty-worktree changes.
- No sudo. Use `.venv` or user-space installs.

## Project Structure

| Path | Purpose |
|---|---|
| `src/` | Trainer, model freezing, normalization, filtering, dedup, scoring |
| `configs/` | Training and search YAML configs |
| `scripts/` | Operational, dataset, LR-search, and documentation tools |
| `benchmark/` | Inference benchmarks and capacity planning |
| `data/` | Training manifests and derived subsets |
| `/home/mahmud/datasets/` | Prepared audio/manifests outside Git |
| `models/` | Promoted/protected model artifacts |
| `outputs_full_gold/` | Active full Gold training outputs |
| `outputs_lr_search/` | LR-search artifacts and metrics |
| `reports/` | Generated audit/search/quality reports |
| `docs/` | Authoritative project memory |

## Development Commands

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m py_compile src/*.py scripts/*.py scripts/lr_search/*.py benchmark/*.py
python -m text_normalization.tests
python scripts/lr_search/audit_data_leakage.py
python scripts/lr_search/verify_freeze_modes.py
git diff --check
```

For non-search training sanity:

```bash
python src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml \
  --sanity-check --sanity-report logs/sanity.json
```

## Coding Style

Use Python 3, four-space indentation, `snake_case` names, `pathlib.Path`, structured
CSV/JSON/YAML parsing, and concise comments only for non-obvious logic. Match local
style; no repository-wide formatter is configured.

## Data and Experiment Integrity

Gold governance schema differs from training schema. Do not interchange them without a
tested conversion. LR-search and active Gold training must keep:

```yaml
load_test_split: false
evaluate_test_after_training: false
```

Every meaningful experiment must record config, logs, metrics, output path, decision,
and failure/lesson if applicable.
