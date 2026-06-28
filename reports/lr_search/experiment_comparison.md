# LR Search Experiment Comparison

Ranking order: validation WER, validation CER, then stability.
Test metrics are forbidden during LR search.

| Rank | Experiment | Trainable layers | Decoder LR | Encoder LR | WER | CER | Runtime h | Stable | Notes |
|---:|---|---|---:|---:|---:|---:|---:|:---:|---|
| 1 | phase3_freeze_boundary_15 | encoder_16_31_plus_decoder | 8e-06 | 5e-06 | 0.236632 | 0.060961 | 7.79 | yes |  |
| 2 | phase2_upper_encoder_8em06 | encoder_24_31_plus_decoder | 8e-06 | 8e-06 | 0.253990 | 0.067542 | 6.64 | yes |  |
| 3 | phase2_upper_encoder_5em06 | encoder_24_31_plus_decoder | 8e-06 | 5e-06 | 0.267957 | 0.072317 | 8.10 | yes |  |
| 4 | phase2_upper_encoder_2em06 | encoder_24_31_plus_decoder | 8e-06 | 2e-06 | 0.283919 | 0.075581 | 6.28 | yes |  |
| 5 | phase2_upper_encoder_1em06 | encoder_24_31_plus_decoder | 8e-06 | 1e-06 | 0.299082 | 0.079492 | 6.32 | yes |  |
| 6 | phase2_upper_encoder_5em07 | encoder_24_31_plus_decoder | 8e-06 | 5e-07 | 0.311652 | 0.082486 | 6.16 | yes |  |
| 7 | phase1b_decoder_lr_8e6 | decoder_only | 8e-06 | 8e-06 | 0.440624 | 0.116689 | 1.50 | yes |  |
| 8 | phase1a_decoder_lr_2e5 | decoder_only | 2e-05 | 8e-06 | 0.443623 | 0.124821 | 0.67 | yes |  |
| 9 | phase2_decoder_8em06 | decoder_only | 8e-06 | 8e-06 | 0.492418 | 0.138403 | 2.85 | yes |  |
| 10 | phase1a_decoder_lr_8e6 | decoder_only | 8e-06 | 8e-06 | 0.497201 | 0.131092 | 1.32 | yes |  |
| 11 | phase1b_decoder_lr_2e5 | decoder_only | 2e-05 | 8e-06 | 0.526589 | 0.139357 | 1.73 | yes |  |
| 12 | phase1b_decoder_lr_2e6 | decoder_only | 2e-06 | 8e-06 | 0.583966 | 0.144087 | 1.49 | yes |  |
| 13 | phase1a_decoder_lr_2e6 | decoder_only | 2e-06 | 8e-06 | 0.644542 | 0.161095 | 1.35 | yes |  |
| 14 | phase1a_decoder_lr_5e5 | decoder_only | 5e-05 | 8e-06 | 2.640144 | 2.144167 | 0.94 | yes |  |
| 15 | phase2_decoder_2em05 | decoder_only | 2e-05 | 8e-06 | 6.313448 | 2.530062 | 1.12 | yes |  |
