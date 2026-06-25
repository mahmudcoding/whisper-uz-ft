# Phase 2 Decoder Confirmation

Candidates were evaluated on the 30h proxy. Tiny differences are treated as noise.
Decision: decoder LR `8e-06`. Lowest validation WER.

| Experiment | Decoder LR | Encoder LR | Mode | WER | CER | Eval loss | Hallucination | Stable | Decision |
|---|---:|---:|---|---:|---:|---:|---:|:---:|---|
| phase2_decoder_8em06 | 8e-06 | 8e-06 | decoder_only | 0.492418 | 0.138403 | 0.489419 | 0.002361 | yes | SELECTED |
| phase2_decoder_2em05 | 2e-05 | 8e-06 | decoder_only | 6.313448 | 2.530062 | 0.435506 | 0.252656 | no | CANDIDATE |

Candidates were evaluated on the 30h proxy. Tiny differences are treated as noise.
Decision: decoder LR `8e-06`. Lowest validation WER.
