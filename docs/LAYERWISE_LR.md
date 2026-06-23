# Layer-Wise Learning Rate Implementation

Generated: 2026-06-23 UTC

## Goal

Use a lower LR for the encoder and a higher LR for the decoder:

- Encoder LR: `2e-6`
- Decoder LR: `8e-6`

This targets Uzbek decoder language-prior adaptation while preserving more of Whisper's acoustic encoder.

## Config

Added support for:

```yaml
encoder_learning_rate: 2.0e-6
decoder_learning_rate: 8.0e-6
```

`learning_rate` remains as fallback/default and scheduler reference.

Best-checkpoint selection is explicit:

```yaml
metric_for_best_model: wer
greater_is_better: false
```

## Implementation

`src/train.py` now constructs explicit PyTorch `AdamW` parameter groups:

- `encoder_decay`
- `encoder_no_decay`
- `decoder_decay`
- `decoder_no_decay`

Grouping rules:

- Parameters under `model.encoder.*` use `encoder_learning_rate`.
- Parameters under `model.decoder.*` and `proj_out.*` use `decoder_learning_rate`.
- Bias and layer norm weights use zero weight decay.
- All trainable parameters are deduplicated by object id before grouping.

The optimizer is passed to `Seq2SeqTrainer` as:

```python
optimizers=(optimizer, None)
```

Trainer still owns scheduler construction, warmup, gradient clipping, checkpointing, and resume behavior.

## Validation

Sanity command:

```bash
cd /home/mahmud/whisper-uz-ft
. .venv/bin/activate
PYTHONPATH=src python src/train.py \
  --config configs/full_ft_dry_run.yaml \
  --sanity-check \
  --sanity-report logs/layerwise_sanity_report.json
```

Result:

- Status: ok
- Device: `cuda:0`
- Trainable params: 1,543,490,560
- Forward loss: `7.58465576171875`
- Peak sanity VRAM: `6115.792 MiB`

Optimizer group report:

- `encoder_decay`: 636,472,320 params, LR `2e-6`, WD `0.03`
- `encoder_no_decay`: 496,640 params, LR `2e-6`, WD `0.0`
- `decoder_decay`: 905,822,720 params, LR `8e-6`, WD `0.03`
- `decoder_no_decay`: 698,880 params, LR `8e-6`, WD `0.0`

Report path:

- `outputs_full_ft_dry_run/optimizer_param_groups.json`

## Metric Fix

Train-time validation emits `eval_wer`, which Trainer maps from `metric_for_best_model: wer`.

Before final test evaluation, `src/train.py` removes the `EarlyStoppingCallback`. This prevents the callback from interpreting post-training `test_wer` as a missing validation `eval_wer`.

## Risk

Encoder LR `2e-6` may be too conservative if Uzbek acoustic adaptation is the real bottleneck. If WER plateaus above partial FT, the next experiment should increase encoder LR to `3e-6` or use layer-wise decay rather than returning to uniform LR immediately.

