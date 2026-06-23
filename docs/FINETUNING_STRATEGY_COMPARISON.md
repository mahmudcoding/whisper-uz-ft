# Fine-Tuning Strategy Comparison

Generated: 2026-06-23 UTC

## Option A: Full Fine-Tuning

All encoder and decoder layers train.

- Rationale: Uzbek-only objective permits overwriting multilingual priors.
- Expected WER impact: highest upside.
- Compute cost: highest, but dry run shows A40 memory is sufficient.
- Risk: overfits 104h clean USC, degrades non-Uzbek languages, can forget general acoustic robustness.

Verdict: best first experiment for Uzbek-only WER/CER.

## Option B: Layer-Wise LR Decay

All layers train, but lower encoder layers receive smaller LR.

- Rationale: preserves low-level acoustic representations while adapting higher abstractions and decoder.
- Expected WER impact: high, possibly more stable than full uniform LR.
- Compute cost: same as full FT.
- Risk: requires optimizer parameter grouping and more implementation complexity.

Verdict: best second experiment if full uniform LR overfits or destabilizes.

## Option C: Decoder-Higher LR Than Encoder

Decoder receives higher LR; encoder receives lower LR.

- Rationale: many Uzbek errors are decoder language-prior errors, including Turkish/Kazakh substitutions.
- Expected WER impact: high for language-prior errors; lower for acoustic/domain mismatch.
- Compute cost: same as full FT.
- Risk: may not adapt lower acoustic/script-alignment layers enough.

Verdict: strong fallback if full uniform FT is too aggressive.

## Option D: Progressive Unfreezing

Start decoder/high encoder, then gradually unfreeze lower encoder layers.

- Rationale: safer if data is small or noisy.
- Expected WER impact: medium-high, but slower to discover optimum.
- Compute cost: multiple phases and more operational complexity.
- Risk: less aggressive than the Uzbek-only objective needs.

Verdict: not preferred for the immediate max-performance experiment.

## Updated Comparison: Uniform LR vs Layer-Wise LR

### A. Uniform LR Full FT

Config: encoder and decoder both use `8e-6`.

Strengths:

- Maximum adaptation pressure.
- Best chance to overwrite all harmful multilingual priors.
- Simpler optimizer behavior.

Weaknesses:

- More likely to damage useful acoustic representations in lower/mid encoder layers.
- Higher overfitting risk on 104h clean USC.
- If Uzbek errors are primarily decoder language-prior errors, uniform encoder updates spend risk where it is less needed.

### B. Layer-Wise LR Full FT

Config: encoder `2e-6`, decoder `8e-6`.

Strengths:

- Directly targets decoder language prior, where Turkish/Kazakh substitution is likely expressed.
- Keeps full model trainable while reducing acoustic drift.
- Lower risk than uniform LR for a first multi-day full-FT run.

Weaknesses:

- If encoder representations are materially bad for Uzbek phonetics, this may adapt too slowly.
- More complex optimizer setup.
- Requires careful logging of optimizer groups for reproducibility.

## Updated Recommendation

Launch layer-wise LR full FT before uniform LR full FT.

Rationale: the observed Uzbek failure mode is heavily language-prior driven, while Whisper's acoustic encoder is already strong. Encoder `2e-6` plus decoder `8e-6` is the better risk-adjusted experiment: it still allows full-model adaptation but puts the strongest update pressure on the decoder.
