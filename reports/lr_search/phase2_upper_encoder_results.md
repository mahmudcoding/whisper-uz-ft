# Phase 2 Upper Encoder Results

Decoder-only is included as the control. Encoder 24-31 is promoted only if validation
evidence beats or materially ties the decoder-only result without stability regression.

| Experiment | Decoder LR | Encoder LR | Mode | WER | CER | Eval loss | Hallucination | Stable | Decision |
|---|---:|---:|---|---:|---:|---:|---:|:---:|---|
| phase2_decoder_8em06 | 8e-06 | 8e-06 | decoder_only | 0.492418 | 0.138403 | 0.489419 | 0.002361 | yes | DECODER-ONLY BASELINE |
| phase2_upper_encoder_5em07 | 8e-06 | 5e-07 | encoder_24_31_plus_decoder | 0.311652 | 0.082486 | 0.343437 | 0.000000 | yes | CANDIDATE |
| phase2_upper_encoder_1em06 | 8e-06 | 1e-06 | encoder_24_31_plus_decoder | 0.299082 | 0.079492 | 0.332487 | 0.000000 | yes | CANDIDATE |
| phase2_upper_encoder_2em06 | 8e-06 | 2e-06 | encoder_24_31_plus_decoder | 0.283919 | 0.075581 | 0.319758 | 0.000000 | yes | CANDIDATE |
| phase2_upper_encoder_5em06 | 8e-06 | 5e-06 | encoder_24_31_plus_decoder | 0.267957 | 0.072317 | 0.300068 | 0.000000 | yes | SELECTED REGIME |

Decoder-only is included as the control. Encoder 24-31 is promoted only if validation
evidence beats or materially ties the decoder-only result without stability regression.
