# Deduplication Strategy

Generated: 2026-06-23 UTC

## Why Aggressive Dedup Matters

Uzbek ASR corpora are likely to share source material across Common Voice mirrors, UzbekVoice derivatives, Saidakmal derivatives, YouTube cuts, and pseudo-labeled collections. Duplicate audio can inflate validation scores and cause memorization. Duplicate text can overweight common proverbs or read prompts.

## Dedup Levels

### 1. Exact Audio Hash

Use `src/dedup/audio_hash.py`.

Method:

- Load audio with `soundfile`.
- Convert to mono float32.
- Quantize to int16 PCM.
- SHA1 hash sample rate + PCM bytes.

Use for:

- Exact duplicate files with different names.
- Mirrored datasets.

### 2. Head Audio Hash + Duration Bucket

Method:

- Hash first 10 seconds.
- Bucket duration to 100ms.

Use for:

- Same recording cut into nearly identical files.
- Re-encoded files with same head content.

Limitation:

- Not robust to heavy compression, gain changes, or leading silence edits.

### 3. Transcript Hash and Near-Duplicate Text

Use `src/dedup/transcript_dedup.py`.

Method:

- Normalize to canonical Uzbek Latin.
- SHA1 hash normalized text.
- Optional `rapidfuzz` near-duplicate check.

Use for:

- Duplicate prompts.
- Repeated read-speech labels.
- Suspicious manifest overlap.

### 4. Dataset Overlap

Use `src/dedup/dataset_overlap.py`.

Signals:

- `duplicate_audio`
- `near_duplicate_audio`
- `text_duration_overlap`
- `overlap_group`

## Recommended Dedup Policy

1. Remove exact duplicate audio globally.
2. For near-duplicate audio, keep the highest-trust sample:
   - Gold beats silver.
   - Silver beats bronze.
   - Higher quality score wins within same tier.
3. For duplicate transcripts with different speakers/audio, do not remove automatically from read-speech corpora; cap sampling weight if overrepresented.
4. For text-duration overlap across datasets, mark suspicious and review before training.
5. Never dedup train/val/test after splitting; dedup before split creation to avoid leakage.

## Execution Order

```bash
PYTHONPATH=src python src/dedup/audio_hash.py \
  --input-csv manifests/combined_raw.csv \
  --output-csv manifests/combined_hashes.csv

PYTHONPATH=src python src/dedup/transcript_dedup.py \
  --input-csv manifests/combined_hashes.csv \
  --output-csv manifests/combined_text_dedup.csv

PYTHONPATH=src python src/dedup/dataset_overlap.py \
  --input-csv manifests/combined_text_dedup.csv \
  --output-csv manifests/combined_overlap.csv
```

