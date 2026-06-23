# Datasets

## Trust Tiers

Gold:

- Human-verified or high-trust transcripts.
- Used for baseline training and clean validation.
- Must be normalized and deduplicated.

Silver:

- Useful and diverse but noisier.
- Must be filtered before training.
- Should be down-weighted relative to Gold.

Bronze:

- Raw or pseudo-labeled.
- Use only after a strong teacher model exists.
- Requires strict filtering and curriculum scheduling.

## Current USC Corpus

Dataset:

- Name: ISSAI Uzbek Speech Corpus.
- Local path: `/home/mahmud/datasets/usc/ISSAI_USC`.
- Project splits: `data/train.csv`, `data/val.csv`, `data/test.csv`.
- Raw samples: 108,387.
- Clean samples: 107,200.
- Hours: `104.63h`.
- Audio: 16 kHz mono WAV.
- Train rows: 99,617.
- Validation rows: 3,762.
- Test rows: 3,821.
- Trust tier: Gold.

Current active full FT uses USC only.

## Gold Master Corpus

Location:

```bash
data/gold_master/
```

Included datasets:

- USC.
- Common Voice Uzbek cleaned subset.
- FLEURS Uzbek.

Blocked:

- FeruzaSpeech.

Final validated size:

- Total rows: 184,140.
- Total hours: `207.12h`.
- Train: 172,135 rows, `186.40h`.
- Validation: 6,068 rows, `10.36h`.
- Test: 5,937 rows, `10.36h`.

Validation:

- Missing audio paths: 0.
- Duplicate path leakage across splits: 0.
- Exact content hash leakage across splits: 0.
- Known speaker leakage across splits: 0.

## Gold Dataset Breakdown

| Dataset | Status | Final rows | Final hours | Known speakers | Trust tier |
|---|---:|---:|---:|---:|---|
| USC | acquired | 107,055 | 104.49h | 936 | Gold |
| Common Voice Uzbek | acquired | 72,917 | 88.54h | 1,366 | Gold |
| FLEURS Uzbek | acquired | 4,168 | 14.08h | 0 reliable | Gold |
| FeruzaSpeech | blocked | 0 | 0h | unknown | Gold candidate |

Common Voice note:

- Official Mozilla Common Voice loader was unavailable in this environment.
- Used accessible cleaned mirror: `yakhyo/mozilla-common-voice-uzbek`.
- Used splits: train, validation, test.
- Excluded raw `validated` and `other`.

FLEURS note:

- Source: `google/fleurs`, config `uz_uz`.
- Speaker IDs are not reliable in exported fields, so FLEURS is split by row rather than speaker.

FeruzaSpeech blocker:

- Source: `k2speech/FeruzaSpeech`.
- Hugging Face status: gated manual.
- Required action: accept terms and configure `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

## Silver Dataset Targets

Planned Silver sources:

- UzbekVoice filtered.
- IT YouTube Uzbek Speech.
- News YouTube Uzbek Speech.
- Podcasts Tashkent Dialect.

Expected value:

- Larger domain diversity than USC.
- Better coverage for production audio.
- Must be deduplicated against Gold and filtered by teacher ASR.

## Bronze Dataset Targets

Planned Bronze sources:

- UzbekVoice raw.
- Saidakmal derivatives.
- Self-collected YouTube.
- Calls and meetings.
- Telephony audio.

Bronze rule:

- Do not train on Bronze until a strong Uzbek teacher checkpoint is available for pseudo-label scoring.

## Dataset Reports

| Report | Path |
|---|---|
| Gold quality summary | `reports/gold_quality_report/summary.json` |
| Gold split validation | `reports/gold_quality_report/master_validation.json` |
| Gold suspicious/rejected rows | `reports/gold_quality_report/rejected_or_suspicious.csv` |
| Dedup summary | `reports/gold_dedup_report/summary.json` |
| Recommended duplicate removals | `reports/gold_dedup_report/duplicate_removals_recommended.csv` |
