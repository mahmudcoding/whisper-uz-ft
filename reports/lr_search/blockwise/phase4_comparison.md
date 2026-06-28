# Phase 4 Blockwise LR Comparison

Ranking order: validation WER, validation CER, then stability. Test metrics are not used.

| Rank | Experiment | A 0-7 | B 8-15 | C 16-23 | D 24-31 | Decoder | WER | CER | Halluc. | Lang conf. | Runtime h | Stable | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | blockwise/phase4b_screen_block_c_5em07_d_8em06 | 0.0 | 0.0 | 5e-07 | 8e-06 | 8e-06 | 0.408836 | 0.108823 | 0.002367 | 0.000000 | 1.61 | yes | stable |
