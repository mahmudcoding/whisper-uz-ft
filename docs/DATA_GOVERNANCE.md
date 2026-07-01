# Data Governance

Last rebuilt: `2026-07-01T04:50:03Z`.

## Data Integrity Rules

- Test data is never used for training, LR search, early stopping, checkpoint selection, or config ranking.
- Stage 1 uses `data/gold_silver_training/train.csv` for training and `data/gold_silver_training/val.csv` for validation. `load_test_split: false` and `evaluate_test_after_training: false` are enforced in the Stage 1 config.
- Gold validation/test remain gold-only and contain zero FeruzaSpeech matches.
- Silver is train-only. It is never used as validation or test.
- Text is normalized to canonical Uzbek Latin before manifests are considered training-ready.
- FeruzaSpeech was moved out of Gold and into Silver because its trust level is lower than USC/Common Voice/FLEURS in the current governance policy.

## Current Manifests

### Gold master governance manifest: data/gold_master
- `train`: 172,135 rows, 186.4037h, source/tier counts: `{'dataset_name': {'usc': 102821, 'common_voice_uz': 67922, 'fleurs_uz': 1392}}`, Feruza matches: 0
- `val`: 6,068 rows, 10.3556h, source/tier counts: `{'dataset_name': {'common_voice_uz': 2467, 'usc': 2203, 'fleurs_uz': 1398}}`, Feruza matches: 0
- `test`: 5,937 rows, 10.3557h, source/tier counts: `{'dataset_name': {'common_voice_uz': 2528, 'usc': 2031, 'fleurs_uz': 1378}}`, Feruza matches: 0

### Gold training schema: data/gold_master_training_schema
- `train`: 172,135 rows, 186.4037h, source/tier counts: `{}`, Feruza matches: 0
- `val`: 6,068 rows, 10.3556h, source/tier counts: `{}`, Feruza matches: 0
- `test`: 5,937 rows, 10.3557h, source/tier counts: `{}`, Feruza matches: 0

### Silver master: data/silver_master
- `train`: 510,702 rows, 795.3530h, source/tier counts: `{'dataset_name': {'uzbekvoice_filtered': 470944, 'feruzaspeech': 12854, 'it_youtube_uz': 11929, 'news_youtube_uz': 10431, 'podcasts_tashkent': 4544}}`, Feruza matches: 12854
- `val`: 0 rows, 0.0000h, source/tier counts: `{'dataset_name': {}}`, Feruza matches: 0
- `test`: 0 rows, 0.0000h, source/tier counts: `{'dataset_name': {}}`, Feruza matches: 0

### Gold+Silver training: data/gold_silver_training
- `train`: 682,837 rows, 981.7567h, source/tier counts: `{'tier': {'silver': 510702, 'gold': 172135}}`, Feruza matches: 12854
- `val`: 6,068 rows, 10.3556h, source/tier counts: `{'tier': {'gold': 6068}}`, Feruza matches: 0
- `test`: 5,937 rows, 10.3557h, source/tier counts: `{'tier': {'gold': 5937}}`, Feruza matches: 0

### LR-search coarse 10h proxy
- `train`: 8,733 rows, 9.9971h, source/tier counts: `{}`, Feruza matches: 0
- `val`: 845 rows, 0.9988h, source/tier counts: `{}`, Feruza matches: 0
- `test`: 818 rows, 0.9992h, source/tier counts: `{}`, Feruza matches: 0

### LR-search main 30h proxy
- `train`: 26,249 rows, 29.9987h, source/tier counts: `{}`, Feruza matches: 0
- `val`: 847 rows, 1.0002h, source/tier counts: `{}`, Feruza matches: 0
- `test`: 816 rows, 1.0002h, source/tier counts: `{}`, Feruza matches: 0

## Silver Preparation State

Silver final unique rows: `510702`, hours: `795.3529802088335`.

Per-dataset Silver composition:

```json
{
  "feruzaspeech": {
    "rows": 12854,
    "hours": 57.82794161458371,
    "speakers": 127,
    "mean_quality_score": 99.48070639489653
  },
  "uzbekvoice_filtered": {
    "rows": 470944,
    "hours": 559.6741800004091,
    "speakers": 4633,
    "mean_quality_score": 100.0
  },
  "it_youtube_uz": {
    "rows": 11929,
    "hours": 81.50745499999994,
    "speakers": 0,
    "mean_quality_score": 99.70450163467181
  },
  "news_youtube_uz": {
    "rows": 10431,
    "hours": 68.89402484375002,
    "speakers": 0,
    "mean_quality_score": 99.6937014667817
  },
  "podcasts_tashkent": {
    "rows": 4544,
    "hours": 27.449378749999994,
    "speakers": 0,
    "mean_quality_score": 99.75572183098592
  }
}
```

Teacher scoring policy:

```json
{
  "teacher_hf_id": "Kotib/uzbek_stt_v1",
  "teacher_revision": "0e239511f65c1c7bbf426619a1ee9ea628411344",
  "teacher_model": "/home/mahmud/models/kotib_uzbek_stt_v1_ct2",
  "teacher_policy": "Kotib Uzbek-only Whisper Medium teacher with forced Uzbek transcription; automatic Whisper language detection is not used as a quality gate",
  "forced_language": "uz",
  "task": "transcribe",
  "parallel_workers": 8,
  "batch_size": 16,
  "input": "/home/mahmud/whisper-uz-ft/data/silver_work/silver_teacher_candidates.csv",
  "output": "/home/mahmud/whisper-uz-ft/data/silver_work/silver_teacher_scored.csv",
  "thresholds": {
    "min_duration_sec": 1.0,
    "max_duration_sec": 30.0,
    "min_chars_per_second": 2.0,
    "max_chars_per_second": 30.0,
    "max_silence_fraction": 0.55,
    "min_snr_proxy_db": 8.0,
    "min_teacher_similarity": 0.82,
    "max_teacher_wer": 0.35,
    "max_teacher_cer": 0.25,
    "min_final_quality_score": 80.0
  }
}
```

Rejection reasons:

```json
{
  "high_teacher_wer": 28966,
  "low_teacher_similarity": 26563,
  "high_teacher_cer": 17489,
  "teacher_error": 16620,
  "text_duration_duplicate_silver": 16562,
  "excessive_silence": 3461,
  "upstream_reported": 3099,
  "low_snr_proxy": 584,
  "duration_too_long": 346,
  "long_utterance": 346,
  "upstream_votes_negative": 278,
  "low_chars_per_second": 258,
  "reading_speed_too_low": 258,
  "suspicious_symbols": 177,
  "transcript_overlap_locked_eval": 57,
  "text_duration_overlap_gold": 13,
  "exact_audio_duplicate_silver": 4
}
```

Dedup report: `reports/silver_dedup_report/summary.json` shows 13 text-duration overlaps with Gold, 57 transcript overlaps with locked eval, 16,562 Silver text-duration duplicates, and 4 exact Silver audio duplicates were handled before final manifests.

## Dataset Storage

- Canonical project manifests live under `data/`.
- Raw/prepared external audio lives under `/home/mahmud/datasets/`.
- Hugging Face dataset cache is not canonical data and may be deleted if it grows.
