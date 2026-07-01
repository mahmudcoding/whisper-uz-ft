# Architecture

Last cleaned for obsolete documentation sections: `2026-07-01T04:52:10Z`.

## Pipeline

Raw/open datasets are staged under `/home/mahmud/datasets/`, normalized and validated by scripts under `scripts/download_datasets/` and `scripts/silver_pipeline/`, then written as CSV manifests under `data/`. Training uses `src/train.py`, model freezing/LR policy lives in `src/model.py`, and evaluation metrics are computed by Hugging Face `evaluate` WER/CER.

## Training Architecture

`src/train.py` loads CSV manifests, disables persistent HF datasets caching for large runs, wraps rows in `OnTheFlySpeechDataset`, computes Whisper features at item load time, pads via `DataCollatorSpeechSeq2SeqWithPadding`, and trains with `Seq2SeqTrainer` plus custom callbacks for JSON metrics, safety checks, production status reports, and best-model snapshots.

## Model Architecture Policy

Current best policy freezes encoder block A (layers 0-7), trains encoder blocks B/C/D (8-31), and trains the decoder. The optimizer uses explicit AdamW parameter groups with frozen parameters excluded.
