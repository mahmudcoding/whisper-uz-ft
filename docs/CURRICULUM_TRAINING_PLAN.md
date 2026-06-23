# Trust-Weighted Curriculum Training Plan

Generated: 2026-06-23 UTC

## Principle

Gold data must never be drowned by noisy data. Sampling is trust-weighted rather than proportional to raw hours.

Default weights:

- Gold: `4.0`
- Silver: `1.5`
- Bronze: `1.0`

Implemented in:

- `src/data_sampling/weighted_sampler.py`

## Stage 1: Gold Only

Datasets:

- USC
- Common Voice Uzbek cleaned subset
- FeruzaSpeech
- FLEURS Uzbek

Expected usable data: ~170-220h.

Training:

- 1-2 epochs.
- Full FT or continue from current full-FT checkpoint.
- Conservative LR if continuing: encoder `1e-6`, decoder `3e-6` to `5e-6`.

Goal:

- Learn clean canonical Uzbek Latin.
- Establish clean-label benchmark.

## Stage 2: Gold + Silver

Datasets:

- Stage 1 data.
- UzbekVoice filtered.
- IT YouTube.
- News YouTube.
- Podcasts Tashkent Dialect.

Expected usable data after filtering: ~600-850h silver plus gold.

Training:

- 1-2 epochs.
- Weighted sampling: gold 4.0, silver 1.5.
- Keep validation gold-only plus a separate robust-dev set.

Goal:

- Robust real-world speech without losing clean Uzbek quality.

## Stage 3: Gold + Silver + Bronze

Datasets:

- Stage 2 data.
- Pseudo labels.
- Calls.
- Meetings.
- Telephony.

Training:

- 1-3 epochs.
- Use bronze only after a strong teacher model exists.
- Filter with teacher WER/CER, transcript similarity, SNR, silence, dedup.

Goal:

- Production robustness across noisy speech and spontaneous speech.

## Stage 4: Domain Adaptation

Datasets:

- Target production domain only.

Training:

- 0.5-1 epoch.
- Low LR.
- Keep gold validation to detect overfit.

Goal:

- Specialize for customer domain without destroying broad Uzbek performance.

