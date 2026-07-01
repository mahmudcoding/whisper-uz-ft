# Disaster Recovery

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

## Recovery Order

1. Read `STATUS.md`, `DATA_GOVERNANCE.md`, `TRAINING_AND_SEARCH.md`, `DECISION_LOG.md`, and `FAILURE_LOG.md`.
2. Recreate environment from `ENVIRONMENT_SETUP.md`.
3. Verify manifests under `data/` and audio under `/home/mahmud/datasets/`.
4. Preserve `models/partial_ft_usc_baseline/` and `outputs_full_gold/best_model/`.
5. Resume training only from complete checkpoints that pass repository validation.

## Stage 1 Resume Command

```bash
cd /home/mahmud/whisper-uz-ft
export PYTHONPATH=/home/mahmud/whisper-uz-ft/src
.venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml --resume auto
```

If no valid checkpoint exists, restart intentionally; do not claim exact resume from a standalone best-model snapshot.
