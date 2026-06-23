# Full Fine-Tuning Plan

Generated: 2026-06-23 UTC

## Implemented Config

Created:

- `configs/full_ft_uzbek.yaml`

Key settings:

| Setting | Value |
| --- | --- |
| Base model | `openai/whisper-large-v3` |
| Trainable layers | all |
| Trainable parameters | 1,543,490,560 |
| Precision | BF16 |
| Epochs | 4 |
| LR | 6e-6 |
| Batch size | 1 |
| Gradient accumulation | 32 |
| Effective batch | 32 |
| Scheduler | cosine |
| Gradient checkpointing | true |
| Eval/save steps | 500 |
| Beam size during eval | 1 |
| Output dir | `outputs_full_ft` |

Sanity check:

- Report: `logs/full_ft_sanity_report.json`
- Status: ok
- CUDA: yes
- Forward loss: 4.5459
- Peak forward-pass VRAM: 6115.8 MiB

## Runtime Estimate

The partial FT run took 10.60h for 1 epoch at 3114 steps. Full FT will have the same number of optimizer steps per epoch but higher backward/optimizer cost.

Estimated A40 runtime:

| Run | Estimate |
| --- | ---: |
| 1 epoch full FT | 14-20h |
| 4 epochs full FT | 56-80h |
| 5 epochs full FT | 70-100h |

## VRAM Estimate

Forward pass uses only about 6.1 GiB, but training memory includes activations, gradients, optimizer states, and checkpointing overhead.

Expected BF16 full FT with batch 1:

- Likely VRAM: 38-46 GiB
- Risk zone: close to A40 48 GB limit

Mitigations:

- Keep batch size 1.
- Keep gradient checkpointing enabled.
- Use BF16, not FP16.
- If OOM: enable optimizer memory reduction, lower eval batch, or use Adafactor/8-bit optimizer after validating stability.

## Launch Command

Do not launch unattended until a 100-300 step dry run is completed.

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
PYTHONPATH=src python src/train.py --config configs/full_ft_uzbek.yaml --sanity-check
```

Then create a short-run config with `max_steps: 200` and run it before the full 4-epoch job.

## Recommendation

Next experiment: full FT BF16, 200-step dry run, then 4 epochs if stable.

Rationale: it directly attacks harmful multilingual priors.

Expected impact: lower Uzbek WER/CER than partial FT if overfitting is controlled.

Risk: read-speech overfitting without diverse data.

