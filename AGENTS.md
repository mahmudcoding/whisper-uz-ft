# Repository Guidelines

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

## Mission

This repository fine-tunes Whisper large-v3 for Uzbek-only ASR. The owner prioritizes minimum Uzbek WER/CER over multilingual retention. Do not optimize for English/Russian/Turkish preservation.

## Read First

1. `docs/STATUS.md`
2. `docs/DATA_GOVERNANCE.md`
3. `docs/TRAINING_AND_SEARCH.md`
4. `docs/DECISION_LOG.md`
5. `docs/FAILURE_LOG.md`
6. `docs/OPERATIONS_RUNBOOK.md`

## Never Do

- Do not delete or modify `models/partial_ft_usc_baseline/`.
- Do not use test data for training, LR search, early stopping, checkpoint selection, or model ranking.
- Do not use this project's own fine-tuned model as the Silver teacher.
- Do not re-enable persistent feature caching for large training runs.
- Do not revert unrelated dirty-worktree changes.

## Best Known Training Strategy

Freeze encoder 0-7; train encoder 8-31 and decoder at `2e-5` with BF16, batch 4, gradient accumulation 8, cosine warmup 10%, duration bucketing, and evaluation/checkpointing every 1000 optimizer steps.

## Project Structure

| Path | Purpose |
|---|---|
| `src/` | Training, model freezing, normalization, filtering, dedup, scoring |
| `configs/` | Training, LR-search, and Stage 1 YAML configs |
| `scripts/` | Dataset, Silver pipeline, LR-search, and operations scripts |
| `benchmark/` | Inference benchmarks and capacity planning |
| `data/` | Canonical manifests and derived training subsets |
| `/home/mahmud/datasets/` | Staged/prepared audio outside Git |
| `outputs_full_gold/` | Completed full-Gold run artifacts |
| `outputs_lr_search/` | LR-search artifacts and metrics |
| `outputs_stage1_gold_silver_nocache/` | Stage 1 Gold+Silver run artifacts |
| `models/` | Protected/promoted model artifacts |
| `reports/` | Generated dataset/search/benchmark reports |
| `docs/` | Authoritative project documentation |
