# Full FT Runtime Optimization

Generated: 2026-06-23 UTC

## Problem

The previous full fine-tuning plan used `eval_steps: 500` and `save_steps: 500`.

Estimated runtime:

- Pure training: about 54-55h for 4 epochs
- Evaluation/checkpoint overhead: about 20-26h
- Total: about 74-80h

The overhead is too high for the first multi-day full-FT experiment.

## Evaluation Cost Model

USC train set: 99,617 samples.

Effective batch:

`1 GPU * batch 1 * grad_accum 32 = 32`

Steps per epoch:

`ceil(99617 / 32) = 3114`

Measured dry-run validation speed:

- 375 validation samples in 314-330s
- About 1.1-1.2 samples/sec

Full validation set:

- 3,762 samples
- Estimated validation runtime: about 53-57 minutes per full validation pass

At `eval_steps=500`, each epoch performs about 6 validation passes. That is roughly 5-6 hours of validation per epoch.

## Options Considered

### Option 1: `eval_steps=1000`, `save_steps=1000`

Validation passes per epoch: about 3.

Expected validation overhead:

- About 2.7-3.0h per epoch
- About 11-12h over 4 epochs

Advantages:

- Reduces overhead by roughly half.
- Still gives visibility inside each epoch.
- Keeps resumable checkpoints reasonably frequent.

Risks:

- Slower detection of divergence or overfitting.
- One failed run loses up to about 4.4h of training progress between checkpoints.

Verdict: recommended.

### Option 2: Epoch-Based Evaluation Only

Validation passes per epoch: 1.

Expected validation overhead:

- About 1h per epoch
- About 4h over 4 epochs

Advantages:

- Lowest overhead.

Risks:

- Too little visibility for a new full-FT recipe.
- Bad LR or overfitting may waste most of an epoch before detection.
- Checkpoints become too sparse for a multi-day run unless save strategy remains step-based.

Verdict: too coarse for the first aggressive full-FT run.

### Option 3: Eval Subset During Training, Full Eval Per Epoch

Advantages:

- Best long-term approach.
- Fast feedback plus reliable full model selection.

Risks:

- Requires trainer changes to maintain two eval datasets and model selection semantics.
- More moving parts before the first full-FT run.

Verdict: implement later after the first layer-wise LR run.

## Applied Change

Updated `configs/full_ft_uzbek.yaml`:

- `eval_steps: 1000`
- `save_steps: 1000`

Estimated new runtime:

- Pure training: about 54-55h
- Validation/checkpoint overhead: about 11-13h
- Total: about 65-68h plus final test

This preserves training visibility while cutting expected overhead by roughly 10-13h.

## Recommendation

Use `eval_steps=1000` and `save_steps=1000` for the next full-FT attempt. Do not switch to epoch-only eval until the layer-wise LR recipe is proven stable.

