# Uzbek Language Prior Enhancement

Generated: 2026-06-23 UTC

## Objective

Aggressively bias the decoder toward canonical Uzbek Latin output. Catastrophic forgetting of other languages is acceptable.

## Methods

### 1. Full Fine-Tuning

Rationale: updates the entire acoustic encoder and decoder toward Uzbek.

Expected impact: highest near-term chance of improving WER/CER over partial FT.

Risk: overfits USC read speech without diverse data.

### 2. Decoder-Heavy Learning Rate

Use parameter groups:

- encoder LR: 3e-6 to 5e-6
- decoder LR: 8e-6 to 1e-5

Rationale: language-prior errors live heavily in the decoder.

Expected impact: faster Uzbek orthographic adaptation.

Risk: decoder can overfit transcripts and hallucinate common Uzbek phrases.

### 3. Layer-Wise LR Decay

Use lower LR for early encoder layers and higher LR for decoder/top encoder.

Rationale: protects low-level acoustics while adapting language-specific representations.

Expected impact: better stability than uniform full FT.

Risk: more training code complexity and harder reproducibility.

### 4. Text-Only Decoder Adaptation

Train decoder-side language behavior with synthetic/no-audio approaches or decoder LM-style adaptation.

Rationale: Uzbek text prior is weak in Whisper.

Expected impact: lower script/word-form errors.

Risk: Whisper decoder is conditioned on audio tokens; pure text adaptation can mismatch inference behavior.

### 5. Shallow Fusion LM / Rescoring

Train an Uzbek Latin LM and rescore ASR beams.

Rationale: external LM can prefer legal Uzbek sequences.

Expected impact: helps word boundaries, suffixes, and common phrase selection.

Risk: slower inference, more deployment complexity, and possible bias toward common text.

## Recommendation

Order of attack:

1. Full BF16 fine-tuning dry run.
2. Full BF16 4-epoch run.
3. Add decoder-heavy LR experiment if full FT helps but plateaus.
4. Add Uzbek LM rescoring only after a strong real-world eval set exists.

