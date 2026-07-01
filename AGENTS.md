# Repository Guidelines

Last rebuilt from repository state: `2026-07-01T04:50:03Z`.

This repository fine-tunes Whisper large-v3 for Uzbek-only ASR. The owner prioritizes minimum Uzbek WER/CER over multilingual retention. Do not optimize for English/Russian/Turkish preservation.

## Immediate Reality

- Active tmux training: `whisper_stage1_gold_silver_nocache`.
- Current config: `configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml`.
- Current output: `outputs_stage1_gold_silver_nocache/`.
- Current logged progress at doc rebuild: step 80/21339 (0.375%).
- GPU is actively used by training: `NVIDIA A40, 100, 38777, 46068, 279.68, 65`.
- Disk is healthy: 2.0T filesystem with ~1.7T free at rebuild.

## Read First

1. `docs/STATUS.md`
2. `docs/DATA_GOVERNANCE.md`
3. `docs/TRAINING_AND_SEARCH.md`
4. `docs/DECISION_LOG.md`
5. `docs/FAILURE_LOG.md`
6. `PROJECT_CONTEXT_EXPORT.txt`

## Never Do

- Do not delete or modify `models/partial_ft_usc_baseline/`.
- Do not use test data for any selection/search/training decision.
- Do not restart or duplicate long training unless the active process is confirmed dead or the user explicitly asks.
- Do not use this project’s own fine-tuned model as Silver teacher.
- Do not re-enable persistent feature caching for large runs.
- Do not trust stale docs over live artifacts.

## Best Known Training Strategy

Freeze encoder 0-7; train encoder 8-31 and decoder at 2e-5 with BF16, batch 4, gradient accumulation 8, cosine warmup 10%, duration bucketing, checkpoint/eval every 1000 steps.

## Current Bottleneck

The active Gold+Silver Stage 1 run must reach validation checkpoints. The next meaningful decision depends on validation WER/CER at step 1000 and later.


See `docs/AGENT_BRIEF.md` and `PROJECT_CONTEXT_EXPORT.txt` for the complete project memory.
