# Decoding Policy

Generated: 2026-06-23 UTC

## Policy

All inference and evaluation must force:

- `language = "uz"`
- `task = "transcribe"`

Automatic language detection is forbidden for this project. The model is Uzbek-only.

## Patched Paths

| File | Status |
| --- | --- |
| `src/model.py` | raises if language/task is not Uzbek/transcribe |
| `src/evaluate_baseline.py` | already forces Uzbek |
| `scripts/transcribe.py` | patched to force Uzbek and avoid ffmpeg |
| `benchmark/eval_suite.py` | language arg restricted to `uz` |
| `benchmark/scripts/benchmark_inference.py` | language/task args restricted to `uz`/`transcribe` |

## Rationale

Raw Whisper large-v3 has harmful multilingual priors on Uzbek speech and can drift into Turkish/Kazakh-like outputs. Forcing Uzbek decoding removes one source of avoidable language confusion.

Expected impact:

- Lower catastrophic language-prior errors.
- More reproducible WER/CER.

Risk:

- Non-Uzbek speech will be forced into Uzbek text. This is acceptable for the current objective.

