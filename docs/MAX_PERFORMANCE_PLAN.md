# Max-Performance Uzbek-Only Training Plan

Generated: 2026-06-23 UTC

## Objective

Minimize Uzbek WER/CER only. Catastrophic forgetting of non-Uzbek languages is acceptable.

## Baseline

Archived immutable baseline:

- Path: `archive/partial_ft_usc/`
- Model: partial FT Whisper large-v3
- Test WER: `0.2005258480`
- Test CER: `0.0529079419`

## Recommended Experiment

Use `configs/full_ft_uzbek.yaml`.

Key settings:

- Full FT, all 1.543B parameters trainable
- BF16
- Batch 1, grad accumulation 32
- Encoder LR `2e-6`
- Decoder LR `8e-6`
- Warmup ratio `0.1`
- Weight decay `0.03`
- Max grad norm `3.0`
- SpecAugment enabled
- 4 epochs with best checkpoint selected by validation WER
- Forced Uzbek transcription for evaluation/inference

## Runtime Estimate

Dataset: 99,617 train samples.

Steps per epoch:

`ceil(99617 / 32) = 3114` optimizer steps.

Measured dry-run training speed:

- Grad accumulation 8: about 3.9-4.0 sec/optimizer step excluding eval.
- Final config grad accumulation 32: estimated 15.6-16.0 sec/optimizer step.

Estimated pure training time:

- Per epoch: about 13.5-13.8 hours
- Four epochs: about 54-55 hours

Evaluation overhead:

- Dry-run validation: 375 samples in 314-330 sec, about 1.1-1.2 samples/sec.
- Full validation: 3,762 samples, roughly 53-57 minutes per validation.
- With eval every 1000 steps: about 3 evals/epoch, adding roughly 2.7-3.0 hours/epoch.

Estimated end-to-end four-epoch runtime with current eval cadence:

- About 65-68 hours, plus final test.

## VRAM and Disk

Measured dry-run peak VRAM:

- About 30.0 GiB allocated on A40 48GB.

Checkpoint storage:

- Each resumable checkpoint: about 18 GiB.
- `save_total_limit: 4`: about 72 GiB retained.
- Final model: about 6 GiB.
- Available disk after dry run: about 1.3 TiB.

Disk is safe.

## Failure Risks

1. Overfitting USC clean read speech.
   - Mitigation: validation WER, best checkpoint selection, compare against archived baseline.
2. Eval overhead dominates wall time.
   - Mitigation: use `eval_steps: 1000` and `save_steps: 1000`; later use eval subset plus epoch-level full eval.
3. Encoder LR too conservative.
   - Mitigation: retry with encoder LR `3e-6` if validation WER plateaus.
4. Decoder LR too aggressive.
   - Mitigation: fallback to decoder LR `5e-6` if validation loss spikes or gradients become unsafe.

## Decision

Do not launch full training until user approval. The current recommendation is layer-wise LR full FT, not uniform LR full FT.
