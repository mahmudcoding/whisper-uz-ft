# Partial FT USC Baseline Report

Generated: 2026-06-23 UTC

## Status

This archive is immutable reference material. Do not modify files under `models/partial_ft_usc_baseline/`.

## Model

- Base model: `openai/whisper-large-v3`
- Fine-tuning strategy: partial fine-tuning
- Frozen: encoder layers 0-23
- Trainable: encoder layers 24-31 and full decoder
- Total parameters: 1,543,490,560
- Trainable parameters: 1,063,930,880
- Trainable percent: 68.93%

## Dataset

- Dataset: ISSAI Uzbek Speech Corpus
- Train: 99,617 samples, 96.14h
- Validation: 3,762 samples, 4.00h
- Test: 3,821 samples, 4.49h
- Audio: 16 kHz mono WAV

## Training Config

- Config: `models/partial_ft_usc_baseline/config/train.yaml`
- Epochs: 1
- Per-device batch: 2
- Gradient accumulation: 16
- Effective batch: 32
- LR: 1e-5
- Scheduler: cosine
- Warmup steps: 1000
- Weight decay: 0.01
- Precision: FP16
- Gradient checkpointing: true
- Eval/save steps: 500
- Beam size during eval: 1

## Results

From `models/partial_ft_usc_baseline/metrics/test_metrics.json`:

- Test loss: 0.2275837064
- Test WER: 0.2005258480
- Test CER: 0.0529079419
- Test runtime: 2802.5382 seconds
- Test samples/sec: 1.363

## Archived Contents

- `model/final_model/`: final saved model
- `metrics/`: test metrics and trainable parameter report
- `checkpoints/`: retained resumable checkpoints
- `logs/`: training and monitor logs
- `config/`: training config and backups

## Evaluation Notes

This is the current best completed baseline. All future full-FT experiments must compare against this exact checkpoint and metrics.
