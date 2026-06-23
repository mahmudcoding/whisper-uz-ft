# Data Normalization Pipeline

Generated: 2026-06-23 UTC

## Canonical Text Target

All training and evaluation text should be normalized to canonical Uzbek Latin before deduplication, quality scoring, and training.

Canonical properties:

- Lowercase.
- Cyrillic converted to Latin.
- Apostrophe variants normalized to `'`.
- `o'` and `g'` represented consistently.
- Unicode normalized with NFKC.
- Punctuation and whitespace cleaned.

Implementation:

- `src/text_normalization/uz_normalizer.py`

## Where Normalization Is Applied

- Existing USC canonicalization: `scripts/download_datasets/prepare_existing_usc.py`
- HF manifest conversion: `scripts/download_datasets/build_manifest_from_hf.py`
- Transcript dedup: `src/dedup/transcript_dedup.py`
- Quality scoring: `src/data_quality/scoring.py`

## Validation Status

Local available datasets:

- USC only.

USC smoke validation:

- Input: `data/mini_test.csv`
- Canonical output: `reports/data_pipeline_smoke/mini_usc_gold.csv`
- Quality output: `reports/data_pipeline_smoke/mini_usc_gold_quality.csv`
- Weighted output: `reports/data_pipeline_smoke/mini_usc_gold_weighted.csv`

Result:

- Rows: 279
- Tier: all `gold`
- Quality classes: 265 `gold`, 14 `bronze`
- Sampling weights observed: `4.0`, `3.4`, `0.7`

## Policy

Never mix raw transcripts from different sources directly. Every dataset must first pass through canonical manifest creation, text normalization, dedup, quality scoring, and weighted sampling.

