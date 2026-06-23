# Gold Corpus Report

Generated: 2026-06-23

## Executive Summary

The Gold Uzbek ASR corpus has been built from the datasets that are locally available or publicly accessible without gated credentials.

Final master corpus:

- Location: `data/gold_master/`
- Train: `data/gold_master/train.csv`
- Validation: `data/gold_master/val.csv`
- Test: `data/gold_master/test.csv`
- Final unique rows: 184,140
- Final unique hours: 207.12h
- Missing audio paths: 0
- Exact audio content leakage across splits: 0
- Known speaker leakage across splits: 0

FeruzaSpeech is not included yet because `k2speech/FeruzaSpeech` is gated manual on Hugging Face and no `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is configured in this environment.

## Dataset Acquisition Status

| Dataset | Status | Local path | Notes |
|---|---:|---|---|
| USC | Complete | `/home/mahmud/datasets/usc/ISSAI_USC` | Existing validated USC corpus prepared into canonical manifests. |
| Common Voice Uzbek | Complete | `/home/mahmud/datasets/common_voice_uz` | Official Mozilla loader was unavailable in this environment; used accessible cleaned mirror `yakhyo/mozilla-common-voice-uzbek`. Only `train`, `validation`, and `test` were used. |
| FLEURS Uzbek | Complete | `/home/mahmud/datasets/fleurs_uz` | Downloaded from `google/fleurs`, config `uz_uz`; exported to 16 kHz mono WAV. |
| FeruzaSpeech | Blocked | `/home/mahmud/datasets/feruzaspeech` | Hugging Face dataset `k2speech/FeruzaSpeech` is gated manual; requires accepted access and an authenticated HF token. |

## Per-Dataset Raw Hours

| Dataset | Raw rows | Raw hours | Known speakers |
|---|---:|---:|---:|
| USC | 107,200 | 104.63h | 936 |
| Common Voice Uzbek | 72,957 | 88.56h | 1,366 |
| FLEURS Uzbek | 4,168 | 14.08h | 0 |
| Total available Gold | 184,325 | 207.27h | 2,302 known |

FLEURS does not expose reliable speaker IDs in the exported dataset, so it is counted as 0 known speakers for speaker-aware split validation.

## Usable Hours After Filtering

Rows rejected by quality checks: 50

Rows removed as exact or near audio duplicates: 135

| Dataset | Final rows | Final usable hours | Keep rows | Suspicious rows |
|---|---:|---:|---:|---:|
| USC | 107,055 | 104.49h | 101,925 | 5,130 |
| Common Voice Uzbek | 72,917 | 88.54h | 71,798 | 1,119 |
| FLEURS Uzbek | 4,168 | 14.08h | 2,995 | 1,173 |
| Total | 184,140 | 207.12h | 176,718 | 7,422 |

Quality decisions are conservative heuristic scores. Suspicious samples are retained in the Gold master for now but should be down-weighted or reviewed before a high-stakes final training run.

## Final Splits

| Split | Rows | Hours | USC rows | Common Voice rows | FLEURS rows |
|---|---:|---:|---:|---:|---:|
| Train | 172,135 | 186.40h | 102,821 | 67,922 | 1,392 |
| Validation | 6,068 | 10.36h | 2,203 | 2,467 | 1,398 |
| Test | 5,937 | 10.36h | 2,031 | 2,528 | 1,378 |

Split validation:

- Missing audio paths: 0
- Duplicate audio path across splits: 0
- Exact content hash leakage across splits: 0
- Known speaker leakage across splits: 0

Validation report: `reports/gold_quality_report/master_validation.json`

## Duplicate Statistics

The dedup pipeline detected and reported multiple duplication signals:

| Signal | Rows flagged |
|---|---:|
| Exact duplicate audio | 260 |
| Near duplicate audio | 262 |
| Removed exact or near audio duplicates | 135 |
| Duplicate transcript | 57,183 |
| Near duplicate transcript | 59,233 |
| Text-duration overlap | 4,492 |

Only exact or near duplicate audio removals were applied automatically. Transcript duplicates were not removed automatically because read-speech corpora naturally contain repeated short phrases and removing them blindly could damage phonetic coverage.

Dedup outputs:

- `reports/gold_dedup_report/summary.json`
- `reports/gold_dedup_report/duplicate_removals_recommended.csv`
- `data/gold_work/gold_hashes.csv`
- `data/gold_work/gold_text_dedup.csv`
- `data/gold_work/gold_overlap.csv`

## Quality Distribution

Across the raw available Gold pool:

| Decision | Rows |
|---|---:|
| Keep | 176,851 |
| Suspicious | 7,424 |
| Reject | 50 |

After rejected rows and audio duplicates were removed, the master corpus contains:

| Split | Keep rows | Suspicious rows |
|---|---:|---:|
| Train | 166,194 | 5,941 |
| Validation | 5,286 | 782 |
| Test | 5,238 | 699 |

Quality report outputs:

- `reports/gold_quality_report/summary.json`
- `reports/gold_quality_report/rejected_or_suspicious.csv`
- `data/gold_work/gold_quality.csv`

## Normalization

All included transcripts were normalized through the project Uzbek normalizer into canonical Uzbek Latin where possible:

- Unicode normalized
- Cyrillic Uzbek converted to Latin
- Apostrophe variants normalized
- Punctuation and whitespace cleaned
- Output manifests use the normalized transcript field

Manifest schema:

- `audio_path`
- `transcript`
- `dataset_name`
- `duration_sec`
- `speaker_id`
- `split`
- `quality_score`

## Remaining Weaknesses

1. FeruzaSpeech is missing.
   - Rationale: gated Hugging Face access blocks automated acquisition.
   - Expected impact of adding it: likely +50-60h of high-value Gold speech diversity.
   - Risk: licensing/access terms must be accepted explicitly by the dataset owner account.

2. FLEURS has no true speaker IDs.
   - Rationale: the exported fields do not provide stable speaker identity.
   - Expected impact: possible speaker leakage inside FLEURS cannot be fully excluded.
   - Risk: evaluation may be slightly optimistic for FLEURS-heavy metrics.

3. Transcript duplicate counts are high.
   - Rationale: read-speech corpora contain many repeated short/common phrases.
   - Expected impact: duplicates may overweight frequent text forms.
   - Risk: aggressive text dedup could remove valid acoustic diversity; keep for now, down-weight later if needed.

4. Quality scoring is heuristic.
   - Rationale: the current Gold pass uses metadata, duration/text checks, normalization checks, audio hash, and audio statistics, not full teacher ASR WER for every sample.
   - Expected impact: obvious bad samples are removed, but subtle label errors may remain.
   - Risk: final SOTA training should add teacher-model agreement scoring once the stronger Uzbek teacher checkpoint is available.

5. Common Voice came from a mirror.
   - Rationale: the official Mozilla Common Voice loader was unavailable in this environment.
   - Expected impact: the mirror provides a practical cleaned Uzbek subset now.
   - Risk: metadata/version may differ from Mozilla official releases; keep source provenance in manifests.

## Next Actions

1. Authenticate Hugging Face and acquire FeruzaSpeech.
2. Re-run Gold build with FeruzaSpeech included.
3. Add teacher ASR scoring after the current full fine-tune produces a strong checkpoint.
4. Use weighted sampling so USC/Common Voice/FLEURS remain balanced and suspicious rows are down-weighted.
5. Keep the current USC-only full fine-tune running; do not switch datasets mid-run.
