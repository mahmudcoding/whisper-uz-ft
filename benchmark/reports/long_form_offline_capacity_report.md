# Long-Form Offline Whisper Capacity Report

Configuration: `faster-whisper`, model `large-v3`, precision `fp16`, USC-derived 5+ hour long-form offline dataset.

Measured runs: 3
GPU-hour cost assumption: $1.2000 for A40 `on_demand` pricing.
Target utilization for cost/audio-hour: 0.70

## Best Throughput Result

- Batch size: `4`
- Beam size: `1`
- End-to-end RTF: `0.0230`
- Processing RTF: `0.0228`
- Throughput: `43.82` audio-hours/hour/GPU
- Cost/audio-hour: `$0.0391`
- Peak VRAM: `5089` MB
- Average GPU utilization: `84.3%`

## Batch And Beam Matrix

| Beam | Batch | Audio h | Startup s | Wall s | RTF e2e | RTF proc | Throughput h/h | Cost/audio h | Avg GPU % | Peak GPU % | Peak VRAM MB | Avg CPU % | P95 latency s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5.00 | 3.7 | 1375.0 | 0.0764 | 0.0762 | 13.13 | $0.1306 | 90.1 | 100.0 | 5505 | 4.9 | 9.7 |
| 1 | 2 | 5.00 | 3.9 | 473.9 | 0.0263 | 0.0261 | 38.30 | $0.0448 | 86.5 | 100.0 | 4545 | 11.0 | 1.7 |
| 1 | 4 | 5.00 | 3.9 | 414.7 | 0.0230 | 0.0228 | 43.82 | $0.0391 | 84.3 | 100.0 | 5089 | 12.3 | 1.7 |

## Offline Capacity Numbers

- Best beam=1 batch: batch `4`, `43.82` audio-hours/hour/GPU.
- Beam=5 was skipped/aborted for this report; no completed full-duration beam=5 result is included.
- Startup overhead is `3.9` seconds on the best run, amortized over `5.00` audio hours.

| Concurrent real-time equivalent streams | Required A40 GPUs at best measured throughput |
|---:|---:|
| 100 | 4 |
| 1,000 | 33 |
| 10,000 | 327 |
| 100,000 | 3,261 |

## Recommendation

For long-form offline throughput on this A40, use batch size `4` with beam size `1` when throughput is the priority.
Use beam size 5 only with decode guardrails such as VAD, max token limits, and hallucination controls; the attempted unguarded beam=5 run was skipped after sustained GPU-bound decoding.
The raw CSV backing this report is `benchmark/reports/long_form_offline_capacity_report.csv`.
