# Language Confusion Benchmark

Generated: 2026-06-23 UTC

## Implemented Tool

Created:

- `benchmark/language_confusion_benchmark.py`

The benchmark wraps `benchmark/eval_suite.py`, forces Uzbek decoding, and flags Turkish/Kazakh/Russian/English leakage in predictions.

Smoke command:

```bash
PYTHONPATH=src:benchmark python benchmark/language_confusion_benchmark.py \
  --model-path outputs/final_model \
  --manifest data/mini_test.csv \
  --max-samples 3 \
  --batch-size 1 \
  --precision fp16 \
  --output reports/language_confusion_smoke.json
```

Smoke result:

| Confusion Type | Rate |
| --- | ---: |
| Turkish-like | 0.0 |
| Kazakh-like | 0.0 |
| Russian Cyrillic | 0.0 |
| English-like | 0.0 |

## Interpretation

Raw Whisper large-v3 had severe language-prior failures on Uzbek. The current final fine-tuned model does not show obvious cross-language leakage in the smoke check, but this benchmark must be run on a larger confusion-focused set.

## Required Dataset

Create a language-confusion test set with:

- Uzbek speech containing words similar to Turkish.
- Uzbek speech containing words similar to Kazakh.
- Uzbek speech with Russian code-switching.
- Uzbek speech with English names and technical terms.
- Dialect speech.

## Risk

The current marker-based classifier is a heuristic. It is useful for regression detection, not for final linguistic claims. For publication-grade reporting, add manual labels or a robust language-ID model on predictions.

