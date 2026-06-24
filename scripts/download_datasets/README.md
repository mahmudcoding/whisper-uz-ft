# Dataset Acquisition Tools

Scripts in this directory stage Uzbek speech datasets under `/home/mahmud/datasets/`
and create manifests for the project data pipeline.

Read [`docs/DATA_GOVERNANCE.md`](../../docs/DATA_GOVERNANCE.md) before acquiring or
exporting data.

## Tools

| Script | Purpose |
|---|---|
| `prepare_existing_usc.py` | convert the local USC corpus into project metadata |
| `download_hf_dataset.py` | download/cache a Hugging Face dataset |
| `export_hf_audio_dataset.py` | export HF audio to local mono 16 kHz files |
| `build_manifest_from_hf.py` | build a manifest from staged HF data |
| `prepare_youtube_dataset.py` | structure already-collected YouTube-style data |

## Rules

- Check license, access, disk, and expected size before large downloads.
- Never commit raw audio or credentials.
- Preserve dataset source, split, speaker, transcript, and duration.
- Normalize transcripts through `src/text_normalization/`.
- Deduplicate against Gold before merging.
- Do not add Silver/Bronze rows directly to training.
- Do not alter validation/test membership during acquisition.

## Environment

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src
export HF_TOKEN='...'  # only when gated access is required
```

Use `--help` on each script before execution. FeruzaSpeech remains gated and must not be
reported as acquired until local export and validation complete.
