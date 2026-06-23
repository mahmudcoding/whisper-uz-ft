# Rubai Pipeline Comparison

Generated: 2026-06-23 UTC

## Source Basis

The local Rubai repository was not present under `/home/mahmud` during this audit. This comparison is based on the prior project analysis notes supplied in the task:

- RubaiSTT v2 uses Whisper Medium.
- They full fine-tune.
- They use BF16.
- They train for 4 epochs.
- They have stronger text normalization.
- They use pseudo-labeling with Gemini.
- They filter corpus data using similarity/WER.
- Their largest advantage is data scale and diversity, about 500h versus our current 104.63h.

Before copying exact code, clone the upstream repository and pin a commit in this document.

## What Rubai Does Better

1. Larger dataset.
   - Rationale: 500h diverse speech is a much stronger signal than 104h clean read speech.
   - Expected impact: large WER reduction on real-world domains.
   - Risk: pseudo-label noise if filtering is weak.

2. Better domain diversity.
   - Includes podcasts, news, conversational audio, and dialect speech.
   - Expected impact: better meetings/calls/podcasts robustness.
   - Risk: licensing and label-quality management.

3. Stronger normalization.
   - Handles Uzbek Cyrillic/Latin mismatch and apostrophe variants.
   - Expected impact: lower measured WER and cleaner outputs.
   - Risk: over-normalization can hide meaningful names or code-switching.

4. Corpus filtering.
   - Uses similarity/WER-style filtering.
   - Expected impact: removes mislabeled or low-quality samples before training.
   - Risk: model-biased filtering can remove valid dialect speech.

## What This Project Does Better

1. Larger base model.
   - Whisper large-v3 has more capacity than Whisper Medium.
   - Expected impact: better ceiling if data quality and diversity improve.
   - Risk: higher training/inference cost.

2. More explicit MLOps structure.
   - Current project has benchmark, monitoring, resume, safety callbacks, and capacity planning.
   - Expected impact: safer production iteration.
   - Risk: engineering overhead if not used consistently.

3. A40 training headroom.
   - Current partial fine-tune used roughly 22.6 GiB peak VRAM.
   - Expected impact: room for BF16, larger batches, or more trainable layers.
   - Risk: full fine-tune still needs careful memory validation.

## Actionable Improvements

1. Integrate the new normalizer into evaluation and data scoring.
2. Build teacher-ASR similarity scoring for USC and new data.
3. Collect 500-1500h of diverse Uzbek audio with explicit licensing.
4. Run BF16 training experiment after normalized evaluation is stable.
5. Compare partial fine-tuning against full fine-tuning on a controlled mini/medium subset.

