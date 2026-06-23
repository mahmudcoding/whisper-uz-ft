# Missing Data Strategy

Generated: 2026-06-23 UTC

## Highest-Priority Gaps

1. Phone calls
2. Meetings
3. Real conversations
4. Dialects outside Tashkent
5. Uzbek-Russian code-switching
6. Elderly speech
7. Children speech

## Collection Plan

### Phone Calls

- Collect consented call-center or simulated telephony.
- Downsample to 8 kHz and 16 kHz variants.
- Preserve channel metadata.
- Label speaker turns if possible.

Impact: very high for enterprise ASR.

Risk: privacy and consent. Must redact PII before training.

### Meetings

- Collect real meeting recordings with overlapping speech, far-field microphones, and pauses.
- Segment with VAD plus speaker-aware chunking.
- Prefer human correction on a 20-50h seed set.

Impact: very high for enterprise transcription.

Risk: diarization/overlap makes labels noisy.

### Real Conversations

- Record interviews, spontaneous dialogues, marketplace/service conversations.
- Balance formal and informal Uzbek.

Impact: high; USC is too read-speech heavy.

Risk: dialect and slang normalization decisions must be consistent.

### Dialects Outside Tashkent

- Target Fergana, Samarkand/Bukhara, Khorezm, Karakalpakstan-adjacent Uzbek, Surkhandarya/Qashqadarya.
- Track dialect metadata.

Impact: high for national robustness.

Risk: over-normalizing dialect words can erase useful variation.

### Uzbek-Russian Code-Switching

- Collect natural mixed Uzbek/Russian speech.
- Do not force Russian words into Uzbek spelling during labels.

Impact: high for business and urban speech.

Risk: current Uzbek-only objective may underfit Russian spans; evaluate separately.

### Elderly and Children Speech

- Purposefully sample age groups.
- Use separate validation buckets.

Impact: medium-high for inclusivity and robustness.

Risk: children speech may require domain adaptation rather than broad training if volume is small.

## Labeling Strategy

1. Human-transcribe gold seed.
2. Pseudo-label with strongest current Whisper large-v3 Uzbek model.
3. Filter with teacher WER/CER and similarity.
4. Human-review disagreement bands.
5. Add only high-confidence bronze to training.

## Privacy Requirements

- Consent records.
- PII redaction.
- Dataset license tracking.
- Source metadata per utterance.

