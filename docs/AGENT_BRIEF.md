# Agent Brief

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

This repository fine-tunes Whisper large-v3 for Uzbek-only ASR. The owner prioritizes minimum Uzbek WER/CER over multilingual retention.

## Read First

1. `docs/STATUS.md`
2. `docs/DATA_GOVERNANCE.md`
3. `docs/TRAINING_AND_SEARCH.md`
4. `docs/DECISION_LOG.md`
5. `docs/FAILURE_LOG.md`
6. `docs/OPERATIONS_RUNBOOK.md`

## Non-Negotiable Rules

- Do not delete or modify `models/partial_ft_usc_baseline/`.
- Do not use test data for any selection/search/training decision.
- Do not use this project's own fine-tuned model as Silver teacher.
- Do not re-enable persistent feature caching for large runs.
- Do not trust stale docs over code/configs/manifests/metrics.

## Best Known Training Strategy

Freeze encoder 0-7; train encoder 8-31 and decoder at `2e-5` with BF16, batch 4, gradient accumulation 8, cosine warmup 10%, duration bucketing, and validation/checkpointing every 1000 optimizer steps.

## Current Experiment Direction

Stage 1 trains Whisper large-v3 on the Gold+Silver mixed corpus using `configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml`. The next model-selection decision depends on validation WER/CER from this run.
