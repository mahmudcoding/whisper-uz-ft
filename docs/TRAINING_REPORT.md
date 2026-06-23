# Training Report

Generated: 2026-06-23 UTC

## Run Status

The full one-epoch partial fine-tuning run completed successfully.

- Launch method: tmux session `whisper_full_training`
- Training command: `python src/train.py --config configs/train.yaml --resume auto`
- Training log: `logs/full_training.log`
- System monitor log: `logs/full_training_system.log`
- Final model: `outputs/final_model`
- Final checkpoint: `outputs/checkpoint-3114`
- Final test metrics: `outputs/test_metrics.json`

The training Python process has exited. The tmux session contains only the monitor window.

## Final Test Metrics

| Metric | Value |
| --- | ---: |
| test_loss | 0.2275837064 |
| test_wer | 0.2005258480 |
| test_cer | 0.0529079419 |
| test_runtime_seconds | 2802.5382 |
| test_samples_per_second | 1.363 |
| test_steps_per_second | 0.682 |

## Validation Curve

| Epoch Fraction | Eval Loss | Eval WER | Eval CER |
| ---: | ---: | ---: | ---: |
| 0.1606 | 0.5178 | 0.4198 | 0.1139 |
| 0.3212 | 0.3903 | 0.3273 | 0.0940 |
| 0.4818 | 0.3062 | 0.2745 | 0.08028 |
| 0.6425 | 0.2640 | 0.2283 | 0.06594 |
| 0.8031 | 0.2393 | 0.2052 | 0.06081 |
| 0.9637 | 0.2311 | 0.2002 | 0.05962 |
| 1.0000 | 0.2309 | 0.2003 | 0.05962 |

The curve is healthy: validation loss, WER, and CER improved monotonically or nearly monotonically through the run. No NaN, OOM, or deadlock was observed in the logs.

## Runtime

| Metric | Value |
| --- | ---: |
| Steps | 3114 |
| Train runtime | 38157.2612 seconds |
| Train runtime | 10.60 hours |
| Train samples/sec | 2.611 |
| Train steps/sec | 0.082 |
| Average step time | 12.26 seconds |
| Final eval runtime | 2499 seconds |
| Final test runtime | 2802.5382 seconds |

## GPU/Memory Health

Peak VRAM from training dry report:

- Peak VRAM: 22,558 MiB

During final test evaluation:

- VRAM: about 24,247 MiB
- GPU utilization: commonly 50-100%
- GPU temperature: low-to-mid 60 C
- Power: roughly 185-246 W

The A40 had large VRAM headroom under the current partial fine-tuning setup.

## Checkpointing and Resume

Resume support is implemented and validated structurally:

- `--resume auto` detects latest checkpoint.
- Checkpoints include model, optimizer, scheduler, scaler, RNG state, trainer state, and training args.
- `CHECKPOINT_OK` was printed for `outputs/checkpoint-3114`.
- `save_total_limit: 3` retained the final three checkpoints.

Resume command:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
PYTHONPATH=src python src/train.py --config configs/train.yaml --resume auto
```

## Recommendation

The current checkpoint is a valid first production-quality Uzbek fine-tune baseline. Do not continue blindly to more epochs on USC alone as the next highest-impact step. The next experiment should add production text normalization into evaluation and compare normalized WER/CER, then train a second epoch or BF16 run only after confirming no real-world regression on meeting/noisy evaluation samples.

## Uzbek-Only Training Addendum

Generated: 2026-06-23 UTC

The recommendation changes under the Uzbek-only objective. Because catastrophic forgetting is acceptable, the next experiment should be full BF16 fine-tuning, not another conservative partial fine-tune.

Validated full-FT setup:

- Config: `configs/full_ft_uzbek.yaml`
- Sanity report: `logs/full_ft_sanity_report.json`
- Trainable parameters: 1,543,490,560 / 1,543,490,560
- Forward pass: ok on CUDA
- Peak sanity-check VRAM: 6115.8 MiB

Required before launch:

- 100-300 step BF16 optimizer dry run.
- Stop immediately on OOM/NaN.
- If stable, run 4 epochs and compare against partial FT on the same normalized USC test set.
