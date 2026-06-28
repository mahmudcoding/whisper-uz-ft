# FeruzaSpeech Gold-to-Silver Migration Report

- Migration UTC: `20260627T112343Z`
- Backup directory: `/home/mahmud/whisper-uz-ft/backups/feruza_gold_to_silver_20260627T112343Z`

## Before

- Gold hours: **264.9430**
- Silver hours: **0.0000**
- Feruza rows in Gold: **12,854** (57.8279h)
- Feruza rows in Silver: **0** (0.0000h)

## After

- Gold hours: **207.1150**
- Silver hours: **57.8279**
- Feruza rows in Gold: **0** (0.0000h)
- Feruza rows in Silver: **12,854** (57.8279h)

## Moved Samples

- Rows moved: **12,854**
- Hours moved: **57.8279**
- Speakers: **127**

## Policy

FeruzaSpeech is removed from Gold and retained as train-only Silver. Original Feruza split is preserved in `source_metadata.original_gold_split` and `data/silver_master/silver_manifest_detailed.csv`.
