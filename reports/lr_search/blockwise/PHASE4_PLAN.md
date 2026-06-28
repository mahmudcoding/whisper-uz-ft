# Phase 4 Blockwise Encoder LR Search Plan

Phase 4 searches depth-dependent encoder learning rates for Whisper large-v3 Uzbek ASR.
Decoder LR is locked at `8e-6`; test manifests remain unloaded and unevaluated.

## Block Definitions

| Block | Layers | Default assumption |
|---|---:|---|
| A | 0-7 | frozen or tiny LR |
| B | 8-15 | very conservative |
| C | 16-23 | conservative adaptation |
| D | 24-31 | strongest encoder adaptation |
| Decoder | 0-31 | locked LR `8e-6` |

Constraint: `LR(D) >= LR(C) >= LR(B) >= LR(A)`.

## Hierarchical Schedule

1. **Phase 4A:** skipped. Block D is selected from the completed upper-encoder search
   plus the queued `phase2_upper_encoder_8em06` add-on.
2. **Phase 4B:** fix D; train C+D+decoder; search C in `5e-7`, `1e-6`, `2e-6`, `5e-6`, `8e-6`.
3. **Phase 4C:** fix C/D; train B+C+D+decoder; search B in `1e-7`, `5e-7`, `1e-6`, `2e-6`.
4. **Phase 4D:** keep A frozen unless B adaptation gives material gains; optional A in `1e-7`, `5e-7`.

Each candidate first runs a 300-step coarse divergence screen. Stable survivors run the
30h proxy. Ranking uses validation WER, then CER, stability, hallucination, and language
confusion. The test split is reserved for final locked-model evaluation only.
