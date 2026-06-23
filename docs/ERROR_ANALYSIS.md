# Error Analysis

Generated: 2026-06-23 UTC

## Inputs

- Full test metrics: `outputs/test_metrics.json`
- Teacher subset: `reports/teacher_subset_final_model_20.json`
- Language confusion smoke: `reports/language_confusion_smoke.json`

## Current Error Profile

The 20-sample teacher subset produced:

- WER: 0.0782
- CER: 0.0139
- RTF: 0.1266
- Peak VRAM: 5465.5 MiB

Observed errors:

1. Acoustic substitutions:
   - `fojia -> fogiya`
   - `qasos yog‘ida qovrildi -> qasos yog‘ida qovurirdi`

2. Word boundary errors:
   - `bir oz -> biroz`
   - `boshida yo‘q` inserted for `boshidayoq`

3. Orthographic variants:
   - `qo‘polligim -> qo‘poligim`
   - `to‘lanishi -> to‘lanchi`

4. No obvious Turkish/Kazakh drift in the final model subset.

## Bottleneck Ranking

1. Data diversity.
   - USC is clean read speech. Enterprise target domains need meetings, calls, podcasts, webinars, and noisy speech.

2. Orthographic normalization.
   - Uzbek apostrophes, joined/split words, Cyrillic/Latin variants, and punctuation strongly affect WER.

3. Acoustic confusions.
   - Current errors are mostly within-Uzbek phonetic/word-boundary mistakes.

4. Decoder language-prior errors.
   - Severe in raw Whisper, much reduced after fine-tuning and forced Uzbek decoding.

5. Russian/English code-switch handling.
   - Not adequately measured by USC. Needs a dedicated benchmark.

## Recommendation

Build a 200-500 sample real-world error set before optimizing further on USC.

Rationale: current USC test errors understate production difficulty.

Expected impact: prevents overfitting to read speech.

Risk: requires manual or high-quality reviewed labels.

