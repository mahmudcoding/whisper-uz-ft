# SILVER Corpus Report

## Final Corpus

- Training rows: **510,702**
- Usable hours: **795.35**
- Evaluation data: Gold validation/test only; no SILVER rows enter model selection or final evaluation.

## Per Dataset

| Dataset | Rows | Hours | Known speakers | Mean quality |
|---|---:|---:|---:|---:|
| feruzaspeech | 12,854 | 57.83 | 127 | 99.48 |
| uzbekvoice_filtered | 470,944 | 559.67 | 4,633 | 100.00 |
| it_youtube_uz | 11,929 | 81.51 | 0 | 99.70 |
| news_youtube_uz | 10,431 | 68.89 | 0 | 99.69 |
| podcasts_tashkent | 4,544 | 27.45 | 0 | 99.76 |

## Rejection Reasons

- `high_teacher_wer`: 28,966
- `low_teacher_similarity`: 26,563
- `high_teacher_cer`: 17,489
- `teacher_error`: 16,620
- `text_duration_duplicate_silver`: 16,562
- `excessive_silence`: 3,461
- `upstream_reported`: 3,099
- `low_snr_proxy`: 584
- `duration_too_long`: 346
- `long_utterance`: 346
- `upstream_votes_negative`: 278
- `low_chars_per_second`: 258
- `reading_speed_too_low`: 258
- `suspicious_symbols`: 177
- `transcript_overlap_locked_eval`: 57
- `text_duration_overlap_gold`: 13
- `exact_audio_duplicate_silver`: 4

## Training Integration

- SILVER governance manifest: `/home/mahmud/whisper-uz-ft/data/silver_master/train.csv`
- Detailed quality manifest: `/home/mahmud/whisper-uz-ft/data/silver_master/silver_manifest_detailed.csv`
- Gold+Silver curriculum manifests: `/home/mahmud/whisper-uz-ft/data/gold_silver_training`
- Initial sampling weights: Gold 4.0, SILVER 1.5, quality-scaled.
