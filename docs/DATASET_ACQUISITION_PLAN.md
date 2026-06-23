# Dataset Acquisition Plan

Generated: 2026-06-23 UTC

## Local Audit

Local project `data/` currently contains only USC-derived manifests:

| Manifest | Rows | Hours | Missing audio |
|---|---:|---:|---:|
| `data/train.csv` | 99,617 | 96.14 | 0 |
| `data/val.csv` | 3,762 | 4.00 | 0 |
| `data/test.csv` | 3,821 | 4.49 | 0 |
| `data/mini_train.csv` | 2,085 | 2.00 | 0 |
| `data/mini_val.csv` | 375 | 0.33 | 0 |
| `data/mini_test.csv` | 279 | 0.33 | 0 |

Audio is staged at `/home/mahmud/datasets/usc/ISSAI_USC`. The wider `/home/mahmud/datasets` tree contains only `usc` at this time.

## Acquisition Inventory

| Tier | Source | Expected usable hours | Local status | Current action |
|---|---|---:|---|---|
| Gold | USC | ~105h | Downloaded, complete, validated | Use now; keep as gold anchor |
| Gold | Common Voice Uzbek cleaned subset | TBD, part of 170-220h gold total | Missing | Prepare HF download; requires license/terms check and cleaning |
| Gold | FeruzaSpeech | ~60h | Missing | Source id/path needed before download; prepare manifest conversion |
| Gold | FLEURS Uzbek | ~10h | Missing | Prepare HF download and canonical manifest |
| Silver | UzbekVoice filtered | TBD, part of 600-850h silver total | Missing | Source id/path needed; do not trust until dedup + teacher scoring |
| Silver | IT YouTube Uzbek Speech | TBD | Missing | Prepare external-manifest ingestion; no scraping without approval |
| Silver | News YouTube Uzbek Speech | TBD | Missing | Prepare external-manifest ingestion; no scraping without approval |
| Silver | Podcasts Tashkent Dialect | TBD | Missing | Prepare external-manifest ingestion; no scraping without approval |
| Bronze | UzbekVoice raw | TBD | Missing | Use only after strong teacher model and strict filtering |
| Bronze | Saidakmal derivatives | TBD | Missing | Treat as overlap-prone; require dedup before use |
| Bronze | Self-collected YouTube | TBD | Missing | Design only; no scraping without approval |
| Bronze | Calls/meetings/telephony | TBD | Missing | Highest production value; collect with consent and privacy controls |

## Scripts Added

Directory: `scripts/download_datasets/`

- `download_hf_dataset.py`: dry-run-first Hugging Face downloader for USC, Common Voice Uzbek, FLEURS Uzbek, and placeholder IDs for Feruza/UzbekVoice.
- `build_manifest_from_hf.py`: converts saved HF datasets to canonical manifests.
- `prepare_existing_usc.py`: tags current USC manifests as gold and normalizes text.
- `prepare_youtube_dataset.py`: converts externally collected YouTube/podcast manifests to canonical schema.
- `README.md`: schema and usage notes.

## Canonical Manifest Schema

All datasets must become:

`audio_path,text,duration,speaker_id,split,source_metadata,dataset_id,tier,trust_weight`

Text must be normalized to canonical Uzbek Latin before quality scoring and training.

## Next Actions

1. Keep current full-FT USC run undisturbed.
2. Acquire missing gold datasets first: Common Voice, FLEURS, FeruzaSpeech.
3. Build a gold-only combined manifest and dedup against USC.
4. Acquire silver only after gold pipeline validates.
5. Run dedup, normalization, quality scoring, and weighted sampling before any new training stage.
