# Data Pipeline

## Pipeline Contract

All datasets must be converted into manifests with stable columns:

```text
audio_path, transcript, dataset_name, duration_sec, speaker_id, split, quality_score
```

Training scripts currently expect USC-style columns:

```text
audio_path, text, duration, speaker_id, split, source_metadata
```

When using `data/gold_master/` for training, either convert column names or update `src/train.py` to accept both schemas.

## Acquisition

Scripts:

- `scripts/download_datasets/export_hf_audio_dataset.py`
- `scripts/download_datasets/prepare_existing_usc.py`
- `scripts/download_datasets/download_hf_dataset.py`
- `scripts/download_datasets/build_manifest_from_hf.py`
- `scripts/download_datasets/prepare_youtube_dataset.py`

Current staged paths:

- USC: `/home/mahmud/datasets/usc/ISSAI_USC`
- Common Voice Uzbek: `/home/mahmud/datasets/common_voice_uz`
- FLEURS Uzbek: `/home/mahmud/datasets/fleurs_uz`

## Audio Validation and Conversion

Requirements:

- Mono.
- 16 kHz.
- WAV or an internally supported audio format.
- Existing export script writes 16 kHz mono WAV for Common Voice and FLEURS.

Validation checks:

- File exists.
- Duration is nonzero.
- Audio can be decoded.
- Sample rate is correct after export.

## Text Normalization

All transcripts must pass through:

```python
from text_normalization import normalize_uzbek_text
```

Normalization converts mixed Uzbek script into canonical Latin where possible and cleans apostrophes, punctuation, Unicode, and whitespace.

## Deduplication

Implemented modules:

- `src/dedup/audio_hash.py`
- `src/dedup/transcript_dedup.py`
- `src/dedup/dataset_overlap.py`

Gold dedup outputs:

- `data/gold_work/gold_hashes.csv`
- `data/gold_work/gold_text_dedup.csv`
- `data/gold_work/gold_overlap.csv`
- `reports/gold_dedup_report/summary.json`

Gold dedup results:

- Exact duplicate audio flagged: 260 rows.
- Near duplicate audio flagged: 262 rows.
- Exact or near audio duplicate removals applied: 135 rows.
- Duplicate transcripts flagged: 57,183 rows.
- Near duplicate transcripts flagged: 59,233 rows.
- Text-duration overlap flagged: 4,492 rows.

Transcript duplicates were not removed automatically because read-speech corpora can contain valid repeated phrases.

## Quality Scoring

Implemented modules:

- `src/data_quality/scoring.py`
- `src/filtering/filter_dataset.py`
- `src/filtering/scoring.py`
- `src/filtering/similarity.py`

Gold scoring results:

- Keep: 176,851 rows.
- Suspicious: 7,424 rows.
- Reject: 50 rows.

Rejected rows and exact/near audio duplicates were removed from the master corpus.

## Split Construction

Gold master outputs:

- `data/gold_master/train.csv`
- `data/gold_master/val.csv`
- `data/gold_master/test.csv`

Rules:

- Avoid exact audio duplicates across splits.
- Avoid known speaker leakage across splits for datasets with true speaker IDs.
- Keep dataset mix in validation and test.
- FLEURS has no true speaker IDs, so its leakage risk cannot be eliminated by speaker-aware splitting.

Validation output:

- `reports/gold_quality_report/master_validation.json`

## Future Improvements

P0 before Gold training:

- Adapt training loader to accept `data/gold_master/` manifest schema.
- Add weighted sampling using `src/data_sampling/weighted_sampler.py`.

P1:

- Add teacher-ASR scoring to quality pipeline.
- Add stricter transcript duplicate down-weighting rather than removal.
- Add per-dataset license metadata columns.
