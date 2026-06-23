# faster-whisper A40 Batch Scaling Report

Configuration: `large-v3`, `faster-whisper`, `fp16`, dataset `smoke`, beam size `1`, offline mode.

Throughput-optimal batch size: `2`.
Best measured RTF: `0.0998`.
Best measured throughput: `10.017` audio-sec/sec (`10.02x` real time).
Best peak VRAM: `4257.0` MB.

| Batch | RTF | Speed x | Throughput audio-sec/s | Avg latency s | P95 latency s | Avg GPU % | Peak GPU % | Avg VRAM MB | Peak VRAM MB | Avg CPU % | Peak CPU % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1183 | 8.45 | 8.450 | 0.376 | 0.587 | 54.4 | 98.0 | 4196.8 | 4257.0 | 10.2 | 18.9 |
| 2 | 0.0998 | 10.02 | 10.017 | 0.317 | 0.477 | 47.9 | 100.0 | 4217.5 | 4257.0 | 11.8 | 19.0 |
| 4 | 0.1038 | 9.64 | 9.638 | 0.330 | 0.483 | 68.5 | 100.0 | 4171.0 | 4257.0 | 11.4 | 18.0 |
| 8 | 0.1029 | 9.72 | 9.721 | 0.327 | 0.504 | 44.3 | 99.0 | 4205.9 | 4257.0 | 11.5 | 17.3 |
| 16 | 0.1032 | 9.69 | 9.692 | 0.328 | 0.479 | 51.8 | 99.0 | 4171.0 | 4257.0 | 11.4 | 18.9 |
| 32 | 0.1025 | 9.76 | 9.757 | 0.326 | 0.503 | 55.9 | 99.0 | 4173.9 | 4257.0 | 11.6 | 19.2 |

## Conclusion

Batch size `2` was throughput-optimal on this A40 smoke run. Larger batches did not improve throughput on this workload and mostly produced similar VRAM/latency profiles, so use batch size 2 as the current A40 default for faster-whisper offline batch inference and re-run on larger 1h benchmark sets before final production sizing.
