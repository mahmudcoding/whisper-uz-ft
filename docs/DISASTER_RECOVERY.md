# Disaster Recovery

Last rebuilt from repository reality: `2026-07-01T04:52:10Z`.

For current live state, read `STATUS.md` first. For full memory transfer, read `../PROJECT_CONTEXT_EXPORT.txt`.

## Recovery Order

1. Read `../PROJECT_CONTEXT_EXPORT.txt`.
2. Read `STATUS.md`, `DATA_GOVERNANCE.md`, `TRAINING_AND_SEARCH.md`, `DECISION_LOG.md`, and `FAILURE_LOG.md`.
3. Recreate environment from `ENVIRONMENT_SETUP.md`.
4. Verify manifests under `data/` and audio under `/home/mahmud/datasets/`.
5. Preserve `models/partial_ft_usc_baseline/` and `outputs_full_gold/best_model/`.
6. Check tmux/PIDs/GPU/disk before resuming or launching training.

## Active Run Resume

For the current Stage 1 run, resume only from valid checkpoints using:

```bash
.venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml --resume auto
```

If no valid checkpoint exists, restart intentionally; do not claim exact resume from a standalone best-model snapshot.
