# Phase 3 Freeze Boundary Results

All three requested regimes are compared on the 30h proxy: decoder-only, the best
encoder 24-31 candidate, and encoder 16-31 using the same encoder/decoder LRs.

| Experiment | Decoder LR | Encoder LR | Mode | WER | CER | Eval loss | Hallucination | Stable | Decision |
|---|---:|---:|---|---:|---:|---:|---:|:---:|---|
| phase2_decoder_8em06 | 8e-06 | 8e-06 | decoder_only | 0.492418 | 0.138403 | 0.489419 | 0.002361 | yes | DECODER-ONLY CONTROL |
| phase2_upper_encoder_5em06 | 8e-06 | 5e-06 | encoder_24_31_plus_decoder | 0.267957 | 0.072317 | 0.300068 | 0.000000 | yes | ENCODER 24-31 CONTROL |
| phase3_freeze_boundary_15 | 8e-06 | 5e-06 | encoder_16_31_plus_decoder | 0.236632 | 0.060961 | 0.267357 | 0.000000 | yes | SELECTED |

All three requested regimes are compared on the 30h proxy: decoder-only, the best
encoder 24-31 candidate, and encoder 16-31 using the same encoder/decoder LRs.
