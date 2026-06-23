# Full Fine-Tuning Config Review

Generated: 2026-06-23 UTC

Reviewed files:

- `configs/full_ft_uzbek.yaml`
- `src/train.py`
- `src/model.py`

## Current Full-FT Config

- Base model: `openai/whisper-large-v3`
- Objective: Uzbek-only ASR
- Trainable layers: all layers
- Precision: BF16
- Epochs: 4
- Batch per device: 1
- Gradient accumulation: 32
- Effective batch: 32
- LR: `8e-6`
- Scheduler: cosine
- Warmup: `0.1` ratio
- Weight decay: `0.03`
- Max grad norm: `3.0`
- Eval/save/logging: 500/500/25 steps
- Save total limit: 4
- Generation: Uzbek transcribe, max length 225, beam 1
- SpecAugment: enabled with Rubai-style light masking

## Review Findings

### Full fine-tuning support

`src/model.py` supports `train_last_encoder_blocks: all` and the dry run confirmed 1,543,490,560 trainable parameters. This is correct for the Uzbek-only objective.

### BF16 support

`src/train.py` now passes `bf16` into `Seq2SeqTrainingArguments`. A40 BF16 training ran successfully in the dry run.

### Learning rate

`8e-6` is aggressive but defensible for full FT because the objective is not multilingual preservation. It matches Rubai's reported LR for Whisper Medium, but large-v3 has more parameters and USC is smaller.

Recommendation: keep `8e-6` for the first full-FT experiment only if validation is frequent and best checkpoint selection is enforced. If validation WER worsens after the first epoch, retry `5e-6`.

### Batch size

Batch 1 with grad accumulation 32 is the right A40-safe choice. The 100-step dry run peaked at about 30.0 GiB VRAM with batch 1 and BF16, leaving enough headroom.

### Eval and save frequency

`eval_steps: 500` gives about 6 validation runs per epoch. Full validation is expensive: dry-run validation decoded at about 1.1-1.2 samples/sec. Full USC validation has 3,762 samples, so each validation pass can take roughly 53-55 minutes.

Recommendation: keep `eval_steps: 500` for safety if running unattended, but understand this adds about 5-6 hours per epoch. For faster experiments, use an eval subset during training and full validation after each epoch.

### Checkpoint storage

Each full optimizer checkpoint is about 18 GiB. `save_total_limit: 4` keeps checkpoint storage around 72 GiB plus final model around 6 GiB. Disk has about 1.3 TiB free, so this is safe.

### Early stopping warning

During the dry run, train-time validation produced `eval_wer` correctly. A warning appeared after post-training `test` evaluation: early stopping expected `eval_wer` but saw `test_wer`. This is not a train-time blocker because early stopping only matters during validation, but it should be cleaned later by disabling/removing the early stopping callback before final `trainer.evaluate(..., metric_key_prefix="test")`.

## Verdict

The config is safe for A40 memory and technically ready for a long run after approval. The main risk is not stability; it is overfitting to clean USC and losing production robustness. That risk is acceptable for the next Uzbek-only experiment but must be measured against the archived partial baseline.

