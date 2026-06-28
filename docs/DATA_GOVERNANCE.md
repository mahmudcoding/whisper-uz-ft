# Data Governance

This file documents the real dataset state, schemas, trust tiers, normalization,
deduplication, leakage policy, and unfinished data work.

## Principles

1. Measure decoded duration; do not trust nominal raw hours.
2. Normalize all transcripts to canonical Uzbek Latin.
3. Keep train, validation, and test separated by path, content hash, and reliable
   speaker IDs where available.
4. Use validation for model selection only.
5. Keep test locked until final evaluation of a selected model.
6. Do not train raw Silver/Bronze data.
7. Gold data must not be drowned by noisier data.
8. Preserve dataset origin, quality score, split, and license/trust notes.

## Dataset Roots

| Path | Role |
|---|---|
| `/home/mahmud/datasets/` | prepared source audio/manifests outside repo |
| `data/gold_master/` | canonical Gold governance manifests |
| `data/gold_master_training_schema/` | Gold converted for `src/train.py` |
| `data/silver_master/` | finalized Silver governance manifests |
| `data/gold_silver_training/` | weighted Gold+Silver training view |
| `data/lr_search/` | deterministic Gold proxy subsets |

## Gold Master

Gold contains USC, Common Voice Uzbek, and FLEURS Uzbek. FeruzaSpeech is not Gold.

| Source | Rows | Hours | Known speakers | Notes |
|---|---:|---:|---:|---|
| USC | 107,055 | 104.4933 | 936 | high-trust read speech |
| Common Voice Uzbek | 72,917 | 88.5389 | 1,366 | cleaned accessible mirror |
| FLEURS Uzbek | 4,168 | 14.0828 | 2 placeholder/limited IDs | speaker isolation limited |
| Total | 184,140 | 207.1150 | - | validated |

Gold split:

| Split | Rows | Hours | USC h | CV h | FLEURS h |
|---|---:|---:|---:|---:|---:|
| train | 172,135 | 186.4037 | 100.5481 | 81.1448 | 4.7109 |
| validation | 6,068 | 10.3556 | 2.0083 | 3.6252 | 4.7221 |
| test | 5,937 | 10.3557 | 1.9369 | 3.7689 | 4.6498 |

Validation report `reports/gold_quality_report/master_validation.json`:

- total rows: 184,140;
- total hours: 207.1150;
- missing audio paths: 0;
- path leakage across splits: 0;
- content-hash leakage across splits: 0;
- known speaker leakage across splits: 0;
- FeruzaSpeech rows: 0.

## Gold Schemas

Governance schema in `data/gold_master/*.csv`:

```text
audio_path,transcript,dataset_name,duration_sec,speaker_id,split,quality_score
```

Training schema in `data/gold_master_training_schema/*.csv`:

```text
audio_path,text,duration,speaker_id,split,source_metadata
```

The training schema does not expose `dataset_name` as a direct CSV column; source data
is encoded in `source_metadata`. Use the governance manifests for per-source audits.

## FeruzaSpeech Migration

FeruzaSpeech was prepared from `/home/mahmud/feruzaspeech.zip` and then moved out of
Gold on 2026-06-27.

Migration report: `reports/silver_quality_report/feruza_gold_to_silver_migration.json`.

Before migration:

- Gold rows: 196,994;
- Gold hours: 264.9430h;
- Feruza rows in Gold: 12,854;
- Feruza hours in Gold: 57.8279h.

After migration:

- Gold rows: 184,140;
- Gold hours: 207.1150h;
- Feruza rows in Gold: 0;
- Silver rows: 12,854;
- Silver hours: 57.8279h.

Reason: gated/restrictive K2Speech terms. The prepared export summary still lists tier
`gold`; that source export is historical. The authoritative project tier is Silver.

## Silver State

Finalized Silver master:

| Source | Rows | Hours | Split | Status |
|---|---:|---:|---|---|
| FeruzaSpeech | 12,854 | 57.8279 | train only | finalized Silver |

`data/gold_silver_training/` is a weighted training view:

| Split | Rows | Hours | Contents |
|---|---:|---:|---|
| train | 184,989 | 244.2317 | Gold train + Feruza train-only Silver |
| validation | 6,068 | 10.3556 | Gold validation only |
| test | 5,937 | 10.3557 | Gold test only |

Weights in the training view:

- Gold trust weight: 4.0;
- Silver trust weight: 1.5;
- Bronze trust weight: 1.0.

Large Silver sources are not finalized:

| Source | Exported raw/prepared hours | Prefilter teacher candidates | Prefilter rejects |
|---|---:|---:|---:|
| UzbekVoice filtered | 596.227h exported | 482,143 rows / 572.473h | 21,235 rows / 23.754h |
| IT YouTube Uzbek | 150.699h exported | 19,718 rows / 142.236h | 1,298 rows / 8.463h |
| News YouTube Uzbek | 149.451h exported | 20,113 rows / 145.568h | 682 rows / 3.883h |
| Tashkent podcasts | 104.119h exported | 13,754 rows / 99.717h | 793 rows / 4.403h |

Teacher scoring state:

- candidate manifest: `data/silver_work/silver_teacher_candidates.csv`;
- candidates: 535,728;
- teacher-scored rows: 129,007;
- teacher: `Kotib/uzbek_stt_v1` revision
  `0e239511f65c1c7bbf426619a1ee9ea628411344`;
- runtime: faster-whisper/CTranslate2 FP16 on CUDA;
- decoding: forced Uzbek, beam 1;
- status: stopped to free GPU for LR search/full Gold training.

Do not use the current project model as Silver teacher.

## Silver Filtering Thresholds

From `reports/silver_quality_report/prefilter_summary.json`:

| Threshold | Value |
|---|---:|
| min duration | 1.0s |
| max duration | 30.0s |
| min chars/sec | 2.0 |
| max chars/sec | 30.0 |
| max silence fraction | 0.55 |
| min SNR proxy | 8.0 dB |
| min language probability | 0.65 |
| min teacher similarity | 0.82 |
| max teacher WER | 0.35 |
| max teacher CER | 0.25 |
| min final quality score | 80.0 |

Silver policy: prioritize precision over size. Reject questionable samples.

## LR-Search Proxy Data

Leakage audit: `reports/lr_search/data_leakage_audit.json`, status `pass`.

| Proxy | Train rows | Train hours | Val rows | Val hours | Test rows | Test hours |
|---|---:|---:|---:|---:|---:|---:|
| coarse_10h | 8,733 | 9.9971 | 845 | 0.9988 | 818 | 0.9992 |
| main_30h | 26,249 | 29.9987 | 847 | 1.0002 | 816 | 1.0002 |

Composition is approximately 50% USC, 40% Common Voice, 10% FLEURS by duration.
FLEURS speaker IDs are not reliable enough for strict speaker-leakage enforcement, but
audio paths and reliable speaker IDs are disjoint.

## Normalization

Implementation:

- `src/text_normalization/uz_normalizer.py`;
- tests: `src/text_normalization/tests.py`.

Canonical behavior:

- Unicode NFKC;
- lowercase;
- Uzbek Cyrillic to Latin;
- apostrophe variants normalized to ASCII `'`;
- `o'` and `g'` canonicalized;
- punctuation and whitespace cleanup.

Important mappings include `ў -> o'`, `қ -> q`, `ғ -> g'`, `ҳ -> h`, `ч -> ch`,
`ш -> sh`, `ю -> yu`, `я -> ya`.

Known limitations: number verbalization is not complete, Russian code-switch spelling
policy is not fully solved, and FLEURS speaker identity is weak.
