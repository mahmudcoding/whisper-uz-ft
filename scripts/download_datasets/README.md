# Dataset Download Scripts

These scripts prepare open Uzbek ASR datasets without silently downloading large corpora.

All large downloads require explicit `--execute`. By default, commands run in dry-run mode and print the intended action.

Target root:

`/home/mahmud/datasets`

Canonical manifest schema:

`audio_path,text,duration,speaker_id,split,source_metadata,dataset_id,tier,trust_weight`

Use `src/text_normalization/uz_normalizer.py` after conversion to normalize text to canonical Uzbek Latin.

Scripts:

- `prepare_existing_usc.py`: tag existing USC CSVs as gold manifests.
- `download_hf_dataset.py`: dry-run-first Hugging Face downloader.
- `build_manifest_from_hf.py`: convert saved HF datasets to canonical manifests.
- `prepare_youtube_dataset.py`: convert externally collected YouTube/podcast/news manifests.
