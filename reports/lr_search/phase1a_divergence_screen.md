# Phase 1A Divergence Screen

Each decoder-only candidate ran for 300 optimizer steps on `coarse_10h`.
Survivors: `[2e-06, 8e-06, 2e-05]`.

| Experiment | Decoder LR | Encoder LR | Mode | WER | CER | Eval loss | Hallucination | Stable | Decision |
|---|---:|---:|---|---:|---:|---:|---:|:---:|---|
| phase1a_decoder_lr_2e6 | 2e-06 | 8e-06 | decoder_only | 0.644542 | 0.161095 | 1.427962 | 0.000000 | yes | SURVIVE |
| phase1a_decoder_lr_8e6 | 8e-06 | 8e-06 | decoder_only | 0.497201 | 0.131092 | 0.582441 | 0.000000 | yes | SURVIVE |
| phase1a_decoder_lr_2e5 | 2e-05 | 8e-06 | decoder_only | 0.443623 | 0.124821 | 0.480980 | 0.001183 | yes | SURVIVE |
| phase1a_decoder_lr_5e5 | 5e-05 | 8e-06 | decoder_only | 2.640144 | 2.144167 | 0.540522 | 0.268639 | no | REJECT: extreme hallucination rate |

Each decoder-only candidate ran for 300 optimizer steps on `coarse_10h`.
Survivors: `[2e-06, 8e-06, 2e-05]`.
