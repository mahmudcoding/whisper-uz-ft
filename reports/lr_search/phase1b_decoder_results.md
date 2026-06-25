# Phase 1B Decoder Results

Full coarse runs use validation metrics only. The top two stable candidates are promoted.
Top-two deltas: WER `0.085966`, CER `0.022668`. Statistical tie thresholds are WER `0.003` and CER `0.001`. Tied: `no`.

| Experiment | Decoder LR | Encoder LR | Mode | WER | CER | Eval loss | Hallucination | Stable | Decision |
|---|---:|---:|---|---:|---:|---:|---:|:---:|---|
| phase1b_decoder_lr_2e6 | 2e-06 | 8e-06 | decoder_only | 0.583966 | 0.144087 | 0.848330 | 0.000000 | yes | ELIGIBLE |
| phase1b_decoder_lr_8e6 | 8e-06 | 8e-06 | decoder_only | 0.440624 | 0.116689 | 0.499464 | 0.000000 | yes | PROMOTE TO PHASE 2 |
| phase1b_decoder_lr_2e5 | 2e-05 | 8e-06 | decoder_only | 0.526589 | 0.139357 | 0.584195 | 0.000000 | yes | PROMOTE TO PHASE 2 |

Full coarse runs use validation metrics only. The top two stable candidates are promoted.
Top-two deltas: WER `0.085966`, CER `0.022668`. Statistical tie thresholds are WER `0.003` and CER `0.001`. Tied: `no`.
