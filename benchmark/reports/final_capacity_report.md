# Whisper Inference Capacity Report

Generated: 2026-06-22T12:47:09.768137+00:00

## Benchmark Summary

- Result file: `/home/mahmud/whisper-uz-ft/benchmark/results/20260622T124700Z_faster-whisper_large-v3_smoke_offline.json`
- Engine: `faster-whisper`
- Model: `large-v3`
- Dataset: `/home/mahmud/whisper-uz-ft/benchmark/datasets/smoke.csv`
- Mode: `offline`
- Precision: `fp16`
- Beam size: `1`
- Batch size: `1`
- Total audio measured: 7.05 seconds
- Processing time: 1.27 seconds
- RTF: 0.1796
- Speed multiplier: 5.57x real time
- Throughput on measured GPU: 5.57 audio-hours/hour
- WER: 1.1666666666666667
- CER: 0.8160919540229885

## Hardware Observed

- GPU: NVIDIA A40
- VRAM: 44.4 GiB
- CPU count: 52
- RAM: 110.0 GiB
- Peak measured VRAM: 4257.0 MB
- Peak GPU utilization: 82.0%

## Pipeline Stages

- Enabled stages: vad, punctuation
- Pipeline latency multiplier: 1.08
- Pipeline cost multiplier: 1.05

## Capacity And Cost

### A40

- Estimated speed multiplier: 5.16x
- Streams/GPU at target utilization: 3.61
- GPU-hour cost: $1.26
- Cost per audio hour: $0.35

| Concurrent streams | GPUs | Servers | Monthly cost | Annual cost |
|---:|---:|---:|---:|---:|
| 100 | 35 | 35 | $32,193.00 | $386,316.00 |
| 1,000 | 347 | 347 | $319,170.60 | $3,830,047.20 |
| 10,000 | 3,464 | 3,464 | $3,186,187.20 | $38,234,246.40 |
| 100,000 | 34,632 | 34,632 | $31,854,513.60 | $382,254,163.20 |

### L4

- Estimated speed multiplier: 2.32x
- Streams/GPU at target utilization: 1.62
- GPU-hour cost: $0.68
- Cost per audio hour: $0.42

| Concurrent streams | GPUs | Servers | Monthly cost | Annual cost |
|---:|---:|---:|---:|---:|
| 100 | 77 | 77 | $38,363.33 | $460,359.90 |
| 1,000 | 770 | 770 | $383,633.25 | $4,603,599.00 |
| 10,000 | 7,696 | 7,696 | $3,834,339.60 | $46,012,075.20 |
| 100,000 | 76,960 | 76,960 | $38,343,396.00 | $460,120,752.00 |

### A100

- Estimated speed multiplier: 11.34x
- Streams/GPU at target utilization: 7.94
- GPU-hour cost: $3.36
- Cost per audio hour: $0.42

| Concurrent streams | GPUs | Servers | Monthly cost | Annual cost |
|---:|---:|---:|---:|---:|
| 100 | 16 | 16 | $39,244.80 | $470,937.60 |
| 1,000 | 158 | 158 | $387,542.40 | $4,650,508.80 |
| 10,000 | 1,575 | 1,575 | $3,863,160.00 | $46,357,920.00 |
| 100,000 | 15,742 | 15,742 | $38,611,977.60 | $463,343,731.20 |

### H100

- Estimated speed multiplier: 19.59x
- Streams/GPU at target utilization: 13.72
- GPU-hour cost: $5.78
- Cost per audio hour: $0.42

| Concurrent streams | GPUs | Servers | Monthly cost | Annual cost |
|---:|---:|---:|---:|---:|
| 100 | 10 | 10 | $42,157.50 | $505,890.00 |
| 1,000 | 92 | 92 | $387,849.00 | $4,654,188.00 |
| 10,000 | 912 | 912 | $3,844,764.00 | $46,137,168.00 |
| 100,000 | 9,114 | 9,114 | $38,422,345.50 | $461,068,146.00 |

## Recommendations

- Best measured engine/precision from available runs: `faster-whisper` / `fp16`.
- Best projected cost per audio hour in this config: `A40` at $0.35.
- Highest projected single-GPU throughput: `H100`.
- For production, prefer faster-whisper/CTranslate2 with fp16 or int8_float16 after converting fine-tuned checkpoints.
- Use offline batching for backlogs and bounded micro-batches for streaming; keep streaming chunk sizes at 5-10s unless latency requirements force 1-2s chunks.
- Re-run the full matrix before purchase decisions; this report uses measured local data plus editable relative hardware multipliers.
- Whisper large-v3 is feasible when accuracy is the primary requirement, but distillation or smaller Whisper variants should be tested for high-concurrency captioning.

