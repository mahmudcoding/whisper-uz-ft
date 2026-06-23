# Hyperparameter Recommendation

Generated: 2026-06-23 UTC

## Search Space Considered

- LR: `3e-6`, `5e-6`, `8e-6`
- Epochs: 2, 3, 4, early stopping
- Warmup: 5%, 10%
- Weight decay: `0.01`, `0.03`
- Strategies: full FT, layer-wise LR decay, decoder-higher LR, progressive unfreezing

## Recommended First Full-FT Experiment

- Full fine-tuning all layers
- BF16
- Per-device batch size: 1
- Gradient accumulation: 32
- Effective batch: 32
- LR: `8e-6`
- Scheduler: cosine
- Warmup ratio: `0.1`
- Weight decay: `0.03`
- Max grad norm: `3.0`
- Epochs: 4 with best-checkpoint selection by validation WER
- SpecAugment: light, Rubai-style
- Generation during eval: language `uz`, task `transcribe`, beam 1

## Rationale

`8e-6` is the highest-risk LR in the search space, but it is justified because the objective is Uzbek-only and the previous partial FT left multilingual priors frozen. Rubai used the same LR with full fine-tuning and BF16. The dry run showed stable loss decline and finite gradients.

10% warmup is preferred over 5% because all 1.54B parameters are trainable and the dataset is only 104h. Weight decay `0.03` is preferred over `0.01` because full FT on USC has overfitting risk.

## Fallbacks

If epoch-1 validation WER is unstable, NaNs appear, or loss spikes:

1. Retry full FT with LR `5e-6`, same batch and regularization.
2. If still unstable, use decoder-higher LR: decoder `8e-6`, encoder `3e-6` to `5e-6`.
3. If overfitting dominates after epoch 1, keep full FT but reduce epochs to 2-3 and scale data before more epochs.

## Expected Impact

- Best case: substantially beats partial FT WER/CER by adapting the whole model away from harmful Turkish/Kazakh priors.
- Moderate case: improves USC test but overfits clean read speech.
- Worst case: underperforms partial FT due to overfitting or too aggressive LR; archived baseline protects against regression.

