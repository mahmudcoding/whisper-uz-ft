# Current Project Status

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

## Project Direction

This project fine-tunes `openai/whisper-large-v3` for Uzbek-only ASR. The optimization target is Uzbek WER/CER. Multilingual retention is not a goal.

## Primary Training Configuration

- Dataset: `data/gold_silver_training/` train split with Gold-only validation.
- Test policy: test split is not loaded during training or model selection.
- Model: `openai/whisper-large-v3`.
- Frozen layers: encoder 0-7.
- Trainable layers: encoder 8-31 and decoder.
- LR: encoder 8-31 `2e-5`, decoder `2e-5`.
- Precision: BF16.
- Batch: per-device batch 4, gradient accumulation 8, effective batch 32.
- Scheduler: cosine with `warmup_ratio: 0.10`.
- Checkpoint policy: save/evaluate every 1000 optimizer steps, retain latest 2 full checkpoints, save best model separately by validation WER.
- Dataset-cache policy: persistent Hugging Face feature caching is disabled; `src/train.py` computes Whisper features on the fly via `OnTheFlySpeechDataset`.

## Best Models

- Protected USC partial-FT baseline: `models/partial_ft_usc_baseline/model/`.
  - Test WER: `20.05%`.
  - Test CER: `5.29%`.
- Best completed Gold-only model: `outputs_full_gold/best_model/`.
  - Validation WER: `14.50%`.
  - Validation CER: `3.67%`.
  - Best step: `5000`.
- Stage 1 Gold+Silver run has no retained validation result yet after the no-cache restart.

## Dataset State

- Gold train: `172,135` rows, `186.4037h`.
- Gold validation: `6,068` rows, `10.3556h`.
- Gold test: `5,937` rows, `10.3557h`.
- Silver train: `510,702` rows, `795.3530h`.
- Gold+Silver train: `682,837` rows, `981.7567h`.
- Gold+Silver validation/test remain Gold-only.
- FeruzaSpeech is classified as Silver, not Gold.

## Important Artifacts

- Stage 1 config: `configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml`.
- Stage 1 output root: `outputs_stage1_gold_silver_nocache/`.
- Full Gold config: `configs/full_training/gold_bcd_decoder_2e5.yaml`.
- Silver reports: `reports/silver_quality_report/summary.json`, `reports/silver_dedup_report/summary.json`.

## Worktree Note

The worktree contains tracked and untracked experiment artifacts. Do not revert or delete unrelated files without explicit instruction.
