# BF16 Feasibility Report

Generated: 2026-06-23 UTC

## Hardware and Runtime

- GPU: NVIDIA A40
- Compute capability: 8.6
- PyTorch: 2.5.1+cu121
- `torch.cuda.is_bf16_supported()`: true

## Actual Model Forward Check

Tested `outputs/final_model` with `WhisperForConditionalGeneration` on CUDA using synthetic large-v3-shaped input.

Generated file:

- `reports/bf16_feasibility.json`

| Precision | Avg Forward Seconds | Peak VRAM MiB | Logits Dtype |
| --- | ---: | ---: | --- |
| fp16 | 0.0557 | 3001.4 | torch.float16 |
| bf16 | 0.0554 | 3001.6 | torch.bfloat16 |

## Recommendation

BF16 is feasible on this A40 and should be tested for the next training experiment.

Rationale:

- Hardware and PyTorch report BF16 support.
- The actual fine-tuned Whisper model completed BF16 forward passes.
- Micro-benchmark speed and VRAM were effectively identical to FP16.
- BF16 generally has better numerical range than FP16 and may reduce instability risk.

Expected impact:

- Similar speed and memory.
- Potentially better stability for longer/full fine-tuning.

Risk:

- This was a forward-pass micro-benchmark, not a full optimizer-step benchmark.
- Next step should run a 100-300 step BF16 training dry run before overnight BF16 training.

Implementation status:

- `src/train.py` now passes `bf16` into `Seq2SeqTrainingArguments`.
- `configs/full_ft_uzbek.yaml` sets `bf16: true` and `fp16: false`.
- Future Uzbek-only training should use BF16 unless the optimizer-step dry run shows instability.
