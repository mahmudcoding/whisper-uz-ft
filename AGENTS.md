# Repository Guidelines

## Purpose and Scope

This repository builds a dedicated, open-weight Uzbek ASR system based on
`openai/whisper-large-v3`. The optimization target is Uzbek WER/CER only.
Catastrophic forgetting of other languages is acceptable when it improves Uzbek.

These instructions apply to the entire repository. More specific `AGENTS.md` files, if
added later, override this file only within their directory tree.

## Mandatory Startup Checklist

Before changing code, configuration, data, or a running experiment:

1. Read `docs/AGENT_BRIEF.md`.
2. Read `docs/STATUS.md`.
3. Read `docs/DECISION_LOG.md`.
4. Read `docs/FAILURE_LOG.md`.
5. Read `docs/DOCUMENTATION_STANDARD.md`.
6. For LR-search work, also read `docs/TRAINING_AND_SEARCH.md` and
   `reports/lr_search/data_leakage_audit.md`.
7. Inspect the real runtime state:

```bash
cd /home/mahmud/whisper-uz-ft
git status --short
tmux ls
pgrep -af 'train.py|run_experiment.py|autonomous_search.py'
nvidia-smi
df -h .
```

Treat the current worktree, process table, checkpoints, logs, and generated metrics as
authoritative. Documentation can lag briefly during an active run; reconcile it with
runtime evidence rather than assuming either source is correct.

## Non-Negotiable Invariants

- Never overwrite, delete, move, or modify `archive/partial_ft_usc/`. It contains the
  best protected baseline: WER `20.05%`, CER `5.29%`.
- Never use test data for training, checkpoint selection, LR search, early stopping, or
  model ranking.
- LR-search runs must keep both:

```yaml
load_test_split: false
evaluate_test_after_training: false
```

- Always force Whisper decoding with `language="uz"` and `task="transcribe"`.
- Do not launch full fine-tuning merely because it is available. One-epoch USC full FT
  underperformed partial FT: WER `22.22%`, CER `5.66%`.
- Do not train raw Silver/Bronze data. Normalize, deduplicate, score, and filter first.
- Do not revert user changes or unrelated dirty-worktree changes.
- Do not interrupt, restart, or duplicate a training job before inspecting its tmux
  session, PID, output directory, checkpoints, and logs.
- Before risky edits to important code/configs, create
  `filename.bak_YYYYMMDDTHHMMSSZ`.
- No sudo. Install dependencies only in `.venv` or user space.
- Never commit tokens, credentials, raw audio, model checkpoints, caches, or generated
  multi-gigabyte artifacts.

## Project Structure

| Path | Responsibility |
|---|---|
| `src/` | Training, model freezing, evaluation, normalization, filtering, dedup, quality scoring, sampling |
| `scripts/` | Operational launch, monitoring, transcription, documentation, and dataset tools |
| `scripts/lr_search/` | Subset creation, leakage audit, experiment runner, controller, comparison |
| `scripts/download_datasets/` | Dataset acquisition and export |
| `configs/` | Main YAML training configurations |
| `configs/lr_search/` | Inherited LR-search experiment configurations |
| `benchmark/` | Inference benchmarks, evaluation suites, telemetry, capacity planning |
| `data/` | Training manifests and derived subsets; not the primary raw-audio store |
| `/home/mahmud/datasets/` | Large local dataset/audio storage outside Git |
| `outputs*/` | Models, checkpoints, trainer state, resolved run configs |
| `logs/` | Training and system-monitor logs |
| `reports/` | Generated audit, quality, evaluation, and search reports |
| `archive/` | Protected historical artifacts and backups |
| `docs/` | Authoritative human/agent project memory |
| `setup/` | Reproducible environment bootstrap and dependency lock |

Do not treat files under `docs/archive/` as current truth.

## Environment Setup

Use the repository virtual environment:

```bash
cd /home/mahmud/whisper-uz-ft
bash setup/install.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

The validated stack uses PyTorch `2.7.1+cu126` on an NVIDIA A40 48 GB. PyTorch must
remain `>=2.6` because Transformers refuses to load optimizer/scheduler checkpoints
with older vulnerable `torch.load` implementations.

Basic environment checks:

```bash
bash scripts/check_env.sh
.venv/bin/pip check
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY
```

## Coding Standards

- Python 3, four-space indentation, UTF-8, and type hints for new public functions.
- Use `snake_case` for functions/files/variables and `PascalCase` for classes.
- Prefer `pathlib.Path` over string path manipulation.
- Use structured parsers and writers for YAML, JSON, JSONL, and CSV.
- Keep configuration in YAML rather than embedding experiment constants in code.
- Preserve backward compatibility with existing configs unless a migration is
  explicitly documented and validated.
- Keep changes narrowly scoped. Avoid unrelated refactors during active experiments.
- Add concise comments only for non-obvious architecture, safety, or statistical logic.
- Deduplicate tied parameters when reporting or constructing optimizer groups.
- Print actionable diagnostics: resolved paths, tuning mode, trainable groups, parameter
  counts, precision, device, resume checkpoint, and failure reason.

There is no repository-wide formatter configured. Match the surrounding style and run
`git diff --check`.

## Validation and Test Commands

Run the checks relevant to the files you changed. The minimum static validation for
training/LR-search changes is:

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m py_compile src/*.py scripts/*.py scripts/lr_search/*.py benchmark/*.py
python -m text_normalization.tests
python scripts/lr_search/validate_lr_subsets.py
python scripts/lr_search/audit_data_leakage.py
python scripts/update_docs.py --check
.venv/bin/pip check
git diff --check
```

For model-freezing changes:

```bash
python scripts/lr_search/verify_freeze_modes.py
```

For a new training configuration:

```bash
python scripts/lr_search/run_experiment.py \
  --config configs/lr_search/<experiment>.yaml \
  --dry-run
```

For non-search training initialization:

```bash
python src/train.py \
  --config configs/<config>.yaml \
  --sanity-check \
  --sanity-report logs/<name>_sanity.json
```

Testing is script-oriented rather than centralized in pytest. Add focused executable
tests near the owning module using `tests.py` or `test_*.py`. A passing import or smoke
test is not sufficient evidence for broad behavior: validate the actual split, config,
checkpoint, CUDA path, or output contract affected by the change.

## Dataset and Benchmark Integrity

Canonical Gold manifests:

- `data/gold_master/train.csv`
- `data/gold_master/val.csv`
- `data/gold_master/test.csv`

LR-search proxies:

- `data/lr_search/coarse_10h/`
- `data/lr_search/main_30h/`

Training-ready LR manifests use:

```text
audio_path,text,duration,speaker_id,split,source_metadata
```

Gold master manifests use:

```text
audio_path,transcript,dataset_name,duration_sec,speaker_id,split,quality_score
```

Do not assume these schemas are interchangeable. Convert explicitly or extend the
loader with tests.

Required integrity rules:

- Train rows feed `train_dataset` only.
- Validation rows feed evaluation, early stopping, and checkpoint selection only.
- Test manifests remain unopened during search.
- Exact audio paths cannot cross splits.
- Reliable speaker IDs cannot cross splits.
- Repeated transcript strings alone are not necessarily leakage in read-speech data.
- Preserve test-manifest hashes recorded in
  `reports/lr_search/data_leakage_audit.md`.
- Do not remove suspicious samples automatically unless the user explicitly approves;
  generate recommendations and reports first.

All text must pass the production normalizer in `src/text_normalization/`. Canonical
output is Uzbek Latin with normalized apostrophes, Unicode, punctuation, and whitespace.

## Training and LR-Search Operations

Main training implementation:

- `src/model.py`
- `src/train.py`

Supported tuning modes:

- `decoder_only`
- `encoder_24_31_plus_decoder`
- `encoder_16_31_plus_decoder`
- full FT through the legacy `train_last_encoder_blocks: all` path

Use BF16 on the A40 unless measured evidence requires otherwise. Keep gradient
checkpointing and `max_grad_norm: 1.0` for large-v3 training.

All LR-search experiments must run through:

```bash
python scripts/lr_search/run_experiment.py --config <config>
```

The autonomous sequence runs through:

```bash
python scripts/lr_search/autonomous_search.py
```

Persistent launch pattern:

```bash
tmux new-session -d -s whisper_lr_search \
  "cd /home/mahmud/whisper-uz-ft && source .venv/bin/activate && \
   export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   python scripts/lr_search/autonomous_search.py \
   2>&1 | tee -a reports/lr_search/autonomous_search_console.log"
```

Before launching, always check whether `whisper_lr_search` already exists. Never run a
second controller against the same output directories.

Monitor without attaching interactively:

```bash
tmux has-session -t whisper_lr_search
tail -f reports/lr_search/autonomous_search_console.log
tail -f reports/lr_search/autonomous_search.log
nvidia-smi
```

Resume a single experiment only with its original experiment ID:

```bash
python scripts/lr_search/run_experiment.py \
  --config configs/lr_search/<config>.yaml \
  --experiment-id <existing-id> \
  --resume auto
```

Do not resume if the resolved config no longer matches the checkpoint’s architecture,
optimizer groups, precision, or dataset.

## Failure Handling

Investigate immediately when any of these occur:

- NaN/Inf loss.
- CUDA OOM.
- Repeated unsafe gradient norms.
- Stale logs combined with idle GPU and no checkpoint activity.
- Missing/corrupt `trainer_state.json`, weights, optimizer, or scheduler state.
- WER worsening for two consecutive evaluations.
- Material hallucination or language-confusion increase.
- Changed test-manifest hash.
- Unexpected test metrics in an LR-search output.

For a recoverable interruption:

1. Confirm the original process is gone.
2. Inspect the latest checkpoint with the repository verifier.
3. Confirm disk capacity.
4. Resume with `--resume auto` and the same experiment ID/config.
5. Record the interruption and resolution in docs/reports.

Do not hide safety failures, bypass `torch.load` protections, or silently restart from
weights while claiming optimizer/scheduler continuity.

## Metrics and Experiment Decisions

Primary model-selection metric: validation WER. Secondary evidence:

- validation CER
- validation loss
- convergence behavior
- hallucination rate
- language-confusion rate
- gradient stability
- consistency across 10h and 30h proxies

WER/CER are stored as ratios (`0.20` means 20%). For LR search, differences below:

- WER `0.003` (0.3 percentage points)
- CER `0.001` (0.1 percentage points)

are practical ties. Do not overfit to tiny numeric differences; prefer the lower LR
when quality and stability are effectively tied.

Never promote a model based only on training loss. Never report test results as
validation results. Final test evaluation occurs once after locking the winning search
configuration.

## Documentation Responsibilities

Documentation is part of the implementation, not optional cleanup. Follow
`docs/DOCUMENTATION_STANDARD.md`.

After meaningful changes:

- Update `docs/STATUS.md` for current jobs/results.
- Append completed experiments to `docs/EXPERIMENT_LEDGER.md`.
- Register decision-relevant models in `docs/MODEL_REGISTRY.md`.
- Record strategy changes in `docs/DECISION_LOG.md`.
- Record failures and fixes in `docs/FAILURE_LOG.md`.
- Update `docs/ROADMAP.md`.
- Keep `docs/AGENT_BRIEF.md` concise and current.
- Update `docs/state.json` when machine-readable state changes.

Do not create redundant top-level docs. Put detailed generated reports under
`reports/` and historical material under `docs/archive/`.

## Git and Review Practices

The repository history uses short imperative subjects such as `rebuild docs`. Use
focused commits with clear verbs, for example:

```text
add lr-search leakage guard
fix decoder-only freeze reporting
document phase1a results
```

Before committing:

```bash
git status --short
git diff --check
git diff --stat
```

Do not stage unrelated user changes. Pull requests should include:

- objective and rationale
- files/configs changed
- exact validation commands
- measured metrics or explicit “not run”
- data-leakage implications
- checkpoint/resume compatibility
- operational rollback instructions

## Definition of Done

A task is complete only when:

1. The requested code/config/data/report exists.
2. Relevant static checks and runtime validations pass.
3. Training or benchmark behavior is verified from actual outputs when required.
4. Dataset/test integrity is proven, not assumed.
5. Active jobs are healthy or intentionally stopped with state preserved.
6. Required reports and authoritative docs are updated.
7. No required process is left in an unknown state.
8. The final response states what changed, what was verified, what is still running,
   and any residual risk.

Do not claim completion from intent, partial logs, or a plausible configuration. Use
current files, process state, checkpoints, and metrics as evidence.
