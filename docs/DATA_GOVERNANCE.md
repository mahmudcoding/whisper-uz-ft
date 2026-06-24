# Data and Governance

**Document role:** Authoritative dataset inventory, schema contracts, normalization,
quality controls, deduplication, and split policy.

## Governance Principles

1. Do not trust raw hour claims; measure decoded audio duration.
2. Normalize every transcript with one canonical Uzbek normalizer.
3. Remove or isolate duplicate audio across datasets and splits.
4. Prevent known speaker leakage across train, validation, and test.
5. Treat validation as model-selection data and test as a locked final benchmark.
6. Preserve dataset origin, trust tier, speaker, split, and quality metadata.
7. Gold data must not be drowned by noisier Silver/Bronze data.
8. Do not delete questionable samples without an auditable recommendation.

## Dataset Inventory

### Gold Master

Location:

```text
data/gold_master/
```

| Dataset | Final rows | Final hours | Known speakers | Status |
|---|---:|---:|---:|---|
| USC | 107,055 | 104.4933 | 936 | acquired |
| Common Voice Uzbek | 72,917 | 88.5389 | 1,366 | acquired |
| FLEURS Uzbek | 4,168 | 14.0828 | unavailable | acquired |
| FeruzaSpeech | 0 | 0 | unknown | blocked by gated HF access |
| **Total** | **184,140** | **207.1150** | - | validated |

Split summary:

| Split | Rows | Hours |
|---|---:|---:|
| train | 172,135 | 186.4037 |
| validation | 6,068 | 10.3556 |
| test | 5,937 | 10.3557 |

Source notes:

- USC: `/home/mahmud/datasets/usc/ISSAI_USC`.
- Common Voice: cleaned accessible mirror
  `yakhyo/mozilla-common-voice-uzbek`; raw `validated` and `other` were excluded.
- FLEURS: `google/fleurs`, config `uz_uz`.
- FeruzaSpeech: `k2speech/FeruzaSpeech`; requires accepted terms and `HF_TOKEN`.

### Trust Tiers

| Tier | Meaning | Training policy |
|---|---|---|
| Gold | Human-verified/high-trust | Primary training and evaluation source |
| Silver | Useful but noisy or partially verified | Filter, deduplicate, and down-weight |
| Bronze | Raw/pseudo-labeled/high uncertainty | Use only with a strong teacher and strict filtering |

Planned Silver: UzbekVoice filtered, IT YouTube, news YouTube, Tashkent podcasts.

Planned Bronze: UzbekVoice raw, Saidakmal derivatives, self-collected media, calls,
meetings, and telephony.

## Manifest Contracts

### Gold Governance Schema

```text
audio_path,transcript,dataset_name,duration_sec,speaker_id,split,quality_score
```

### Training Schema

```text
audio_path,text,duration,speaker_id,split,source_metadata
```

The schemas are not interchangeable. Convert explicitly or extend `src/train.py` with
tests. LR subset creation performs this conversion.

## Acquisition and Export

Scripts:

- `scripts/download_datasets/prepare_existing_usc.py`
- `scripts/download_datasets/download_hf_dataset.py`
- `scripts/download_datasets/export_hf_audio_dataset.py`
- `scripts/download_datasets/build_manifest_from_hf.py`
- `scripts/download_datasets/prepare_youtube_dataset.py`

Audio requirements:

- mono;
- 16 kHz;
- decodable WAV or supported internal format;
- nonzero duration;
- stable absolute path.

Do not start a large download without checking license/access, available disk, and
expected extracted size.

## Text Normalization

Implementation:

- `src/text_normalization/uz_normalizer.py`
- `src/text_normalization/tests.py`

Default behavior:

- Unicode NFKC;
- lowercase;
- apostrophe variants to ASCII `'`;
- Uzbek Cyrillic to canonical Latin;
- punctuation normalization;
- whitespace cleanup;
- repeated apostrophe collapse.

Selected mappings:

| Cyrillic | Latin |
|---|---|
| `ў` | `o'` |
| `қ` | `q` |
| `ғ` | `g'` |
| `ҳ` | `h` |
| `ч` | `ch` |
| `ш` | `sh` |
| `ю` | `yu` |
| `я` | `ya` |

Run tests:

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m text_normalization.tests
```

Known limitations:

- contextual Cyrillic `е` is transliterated simply;
- number verbalization is not implemented;
- full Russian transliteration is not the goal;
- mixed code-switch text requires human/teacher evaluation.

## Deduplication

Modules:

- `src/dedup/audio_hash.py`
- `src/dedup/transcript_dedup.py`
- `src/dedup/dataset_overlap.py`

Gold results:

| Signal | Rows |
|---|---:|
| duplicate audio flagged | 260 |
| near-duplicate audio flagged | 262 |
| audio duplicate/near-duplicate rows removed | 135 |
| duplicate transcript rows flagged | 57,183 |
| near-duplicate transcript rows flagged | 59,233 |
| text-duration overlaps flagged | 4,492 |

Repeated text is not automatically removed because multiple speakers may legitimately
read the same sentence.

## Quality Scoring

Modules:

- `src/data_quality/scoring.py`
- `src/filtering/filter_dataset.py`
- `src/filtering/scoring.py`
- `src/filtering/similarity.py`

Gold decisions before dedup removal:

| Decision | Rows |
|---|---:|
| keep | 176,851 |
| suspicious | 7,424 |
| reject | 50 |

Current scoring is mostly heuristic. Teacher-ASR agreement across the full Gold corpus
is pending and remains a significant quality improvement opportunity.

## Split Integrity

Gold master validation:

- missing audio paths: 0;
- exact path leakage: 0;
- exact content-hash leakage: 0;
- known speaker leakage: 0.

FLEURS lacks reliable speaker IDs; speaker isolation cannot be proven for that source.
This is a documented residual risk, not evidence of known leakage.

Reports:

- `reports/gold_quality_report/summary.json`
- `reports/gold_quality_report/master_validation.json`
- `reports/gold_dedup_report/summary.json`

## LR-Search Proxies

| Proxy | Train hours | Validation hours | Test hours | Train rows |
|---|---:|---:|---:|---:|
| coarse | 9.9971 | 0.9988 | 0.9992 | 8,733 |
| main | 29.9987 | 1.0002 | 1.0002 | 26,249 |

Training composition is approximately 50% USC, 40% Common Voice, 10% FLEURS by
duration. Sampling is deterministic and stratified by duration and transcript length.

Integrity commands:

```bash
python scripts/lr_search/validate_lr_subsets.py
python scripts/lr_search/audit_data_leakage.py
```

LR-search configs must keep:

```yaml
load_test_split: false
evaluate_test_after_training: false
```

The immutable test hashes are recorded in
`reports/lr_search/data_leakage_audit.md`.

## Weighted Sampling

`src/data_sampling/weighted_sampler.py` defines initial trust weights:

- Gold: 4.0
- Silver: 1.5
- Bronze: 1.0

The helper currently adds row weights to manifests; integration into Trainer sampling
must be verified before relying on it for curriculum training.

## Required Missing Domains

Highest-value gaps:

1. phone calls and telephony;
2. meetings and interruptions;
3. spontaneous conversation;
4. dialects outside Tashkent;
5. Uzbek-Russian code-switching;
6. elderly speech;
7. children speech;
8. noisy far-field audio.

New data must pass the same normalization, deduplication, scoring, and split-governance
pipeline before training.
