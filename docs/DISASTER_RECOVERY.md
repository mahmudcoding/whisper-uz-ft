# Disaster Recovery

**Document role:** Rebuild the project when code, environment, datasets, or artifacts
are lost. This guide assumes the documentation and source repository can be recovered.

## Recovery Priorities

1. Protect or restore the best model archive.
2. Restore source/configs and exact environment.
3. Restore raw datasets and manifests.
4. Validate split integrity.
5. Restore or regenerate model artifacts.
6. Resume only from verified compatible checkpoints.

## Required External Backups

The following cannot be reconstructed from Git alone:

- `models/partial_ft_usc_baseline/`;
- `/home/mahmud/datasets/`;
- large `outputs*` checkpoints/models;
- gated dataset credentials/access;
- any unpublished collected audio.

Maintain independent storage snapshots for these assets.

## 1. Restore Repository

```bash
mkdir -p /home/mahmud
cd /home/mahmud
git clone <repository-url> whisper-uz-ft
cd whisper-uz-ft
```

Read:

- `AGENTS.md`;
- `docs/README.md`;
- `docs/STATUS.md`;
- `docs/DECISION_LOG.md`;
- `docs/FAILURE_LOG.md`.

## 2. Restore Environment

```bash
cd /home/mahmud/whisper-uz-ft
bash setup/install.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

Verify:

```bash
bash scripts/check_env.sh
.venv/bin/pip check
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

Expected critical versions are documented in `ENVIRONMENT_SETUP.md`.

## 3. Restore Raw Datasets

Target root:

```text
/home/mahmud/datasets/
```

Required current Gold sources:

- USC: `/home/mahmud/datasets/usc/ISSAI_USC`;
- Common Voice Uzbek: `/home/mahmud/datasets/common_voice_uz`;
- FLEURS Uzbek: `/home/mahmud/datasets/fleurs_uz`;
- FeruzaSpeech prepared: `/home/mahmud/datasets/feruzaspeech`.

FeruzaSpeech is governed by gated K2Speech terms for academic research/internal use.
Do not redistribute it and verify rights before commercial use. The original ZIP was
deleted after preparation; reacquisition requires renewed gated access.

Use `scripts/download_datasets/` and `DATA_GOVERNANCE.md` for acquisition/export.

## 4. Restore or Rebuild Manifests

Preferred: restore manifests from version control or rebuild them from prepared audio.

Required:

- `data/train.csv`, `data/val.csv`, `data/test.csv`;
- `data/gold_master/train.csv`;
- `data/gold_master/val.csv`;
- `data/gold_master/test.csv`.
- `data/gold_master_training_schema/train.csv`;
- `data/gold_master_training_schema/val.csv`;
- `data/gold_master_training_schema/test.csv`.

Validate paths, durations, source counts, hashes, and speakers against
`DATA_GOVERNANCE.md`.

For LR proxies:

```bash
python scripts/lr_search/create_lr_subsets.py
python scripts/lr_search/validate_lr_subsets.py
python scripts/lr_search/audit_data_leakage.py
```

Expected coarse/main sizes and immutable test hashes are recorded in the leakage audit
and `state.json`.

## 5. Restore Protected Baseline

Restore:

```text
models/partial_ft_usc_baseline/
```

Verify:

```bash
test -d models/partial_ft_usc_baseline/model
python -m json.tool models/partial_ft_usc_baseline/metrics/test_metrics.json
du -sh models/partial_ft_usc_baseline
```

Expected metrics:

- WER `0.2005258480`;
- CER `0.0529079419`.

Do not reconstruct this model by copying an uncertain output directory and labeling it
as the baseline. Provenance must be preserved.

## 6. Validate Source

```bash
python -m py_compile src/*.py scripts/*.py scripts/lr_search/*.py benchmark/*.py
python -m text_normalization.tests
python scripts/lr_search/verify_freeze_modes.py
python scripts/update_docs.py --check
git diff --check
```

## 7. Restore Completed Full-FT Artifacts

If backed up, restore:

- `outputs_full_ft/test_metrics.json` (the non-promoted model weights are not retained);
- `outputs_full_ft/checkpoint-3114/`;
- `outputs_full_ft/test_metrics.json`.

These are useful controls but are not the best model.

If absent, do not rerun full FT solely to restore them. The experiment is documented and
can be reproduced only if needed for analysis.

## 8. Resume Active Full Gold Training

Restore `outputs_full_gold/` if available. Confirm the config matches
`configs/full_training/gold_bcd_decoder_2e5.yaml` and that no duplicate process is
running:

```bash
tmux ls
pgrep -af 'src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml'
```

Resume:

```bash
tmux new-session -d -s whisper_gold_ft \
  "cd /home/mahmud/whisper-uz-ft && \
   export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   .venv/bin/python src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml --resume auto \
   2>&1 | tee -a outputs_full_gold/logs/full_gold_training.log"
```

## 9. Resume Historical LR Search

Restore any completed experiment directories under `outputs_lr_search/`. Confirm each
contains a resolved config and `metrics.json`.

Check no process is running:

```bash
tmux ls
pgrep -af 'autonomous_search.py|run_experiment.py|src/train.py'
```

Then launch:

```bash
tmux new-session -d -s whisper_lr_search \
  "cd /home/mahmud/whisper-uz-ft && source .venv/bin/activate && \
   export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   python scripts/lr_search/autonomous_search.py \
   2>&1 | tee -a reports/lr_search/autonomous_search_console.log"
```

The controller reuses completed metrics and resumes checkpointed runs.

## 10. Reproduce the Partial-FT Baseline

Only if the protected archive is irrecoverably lost:

1. restore exact USC splits;
2. use `configs/train.yaml`;
3. verify encoder 0-23 frozen and 24-31 + decoder trainable;
4. use FP16, one epoch, effective batch 32, LR `1e-5`;
5. train with checkpointing;
6. evaluate the same locked USC test;
7. compare to expected WER/CER;
8. archive with full provenance.

Matching metrics are not guaranteed due to nondeterministic kernels and environment
drift; record any deviation.

## 11. Gold Training Recovery

Before a full Gold run:

1. confirm the selected LR-search evidence and `docs/TRAINING_AND_SEARCH.md`;
2. use `configs/full_training/gold_bcd_decoder_2e5.yaml` unless a newer documented
   winner exists;
3. confirm Gold manifests are converted to the training schema;
4. run leakage and audio validation;
5. run one forward pass;
6. launch in tmux with a unique output directory;
7. run system monitoring;
8. verify the first checkpoint can be loaded with optimizer/scheduler state.

## Recovery Acceptance Test

Recovery is complete when:

- environment checks pass;
- Gold and USC manifests match expected counts/hours;
- test integrity passes;
- protected baseline is present and readable;
- model freeze verification passes;
- one forward pass succeeds on CUDA;
- documentation validation passes;
- any resumed job advances and writes a new healthy log/checkpoint.
