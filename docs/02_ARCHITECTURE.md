# Architecture

## High-Level Flow

```text
Raw datasets
  -> acquisition/staging
  -> audio validation and conversion
  -> Uzbek text normalization
  -> manifest creation
  -> deduplication
  -> quality scoring
  -> train/val/test split creation
  -> Whisper fine-tuning
  -> evaluation and language-confusion analysis
  -> inference and capacity benchmarking
```

## Main Components

### Dataset Acquisition

Scripts:

- `scripts/download_datasets/export_hf_audio_dataset.py`
- `scripts/download_datasets/prepare_existing_usc.py`
- `scripts/download_datasets/download_hf_dataset.py`
- `scripts/download_datasets/build_manifest_from_hf.py`
- `scripts/download_datasets/prepare_youtube_dataset.py`

Current staged datasets live under `/home/mahmud/datasets/`.

### Text Normalization

Code:

- `src/text_normalization/uz_normalizer.py`
- `src/text_normalization/tests.py`

Purpose:

- Convert Uzbek Cyrillic to canonical Uzbek Latin.
- Normalize apostrophe variants.
- Normalize Unicode and punctuation.
- Clean whitespace.
- Make one spoken phrase map to one canonical written form where possible.

### Deduplication

Code:

- `src/dedup/audio_hash.py`
- `src/dedup/transcript_dedup.py`
- `src/dedup/dataset_overlap.py`

Dedup signals:

- Exact audio hash.
- Near audio hash.
- Duration similarity.
- Transcript duplicates.
- Near transcript duplicates.
- Text-duration overlap.

### Quality Scoring

Code:

- `src/data_quality/scoring.py`
- `src/filtering/filter_dataset.py`
- `src/filtering/scoring.py`
- `src/filtering/similarity.py`

Scoring features:

- Transcript length.
- Duration.
- Characters per second.
- Suspicious symbols.
- Empty transcript detection.
- Audio statistics.
- Optional teacher ASR metrics.

### Training

Code:

- `src/train.py`
- `src/model.py`
- `scripts/start_training.sh`
- `scripts/start_mini_training.sh`
- `scripts/start_full_training.sh`
- `scripts/guard_one_epoch_resume.sh`
- `scripts/system_monitor.sh`

Training modes:

- Mini smoke fine-tuning.
- Partial fine-tuning.
- Full Uzbek-only fine-tuning.

### Evaluation

Code:

- `benchmark/eval_suite.py`
- `benchmark/language_confusion_benchmark.py`
- `src/evaluate_baseline.py`

Metrics:

- WER.
- CER.
- Normalized WER/CER where normalization is applied.
- Language-confusion rate.
- Hallucination indicators.
- Inference speed and VRAM.

### Inference Benchmarking

Code:

- `benchmark/scripts/benchmark_inference.py`
- `benchmark/scripts/run_benchmark.sh`
- `benchmark/scripts/capacity_planner.py`
- `benchmark/scripts/create_benchmark_datasets.py`
- `benchmark/scripts/create_long_form_offline_dataset.py`
- `benchmark/scripts/generate_long_form_offline_report.py`

Engines:

- Hugging Face Transformers.
- faster-whisper.
- CTranslate2 conversion script.

## State and Artifact Locations

| Artifact | Path |
|---|---|
| Active config | `configs/full_ft_uzbek.yaml` |
| USC splits | `data/train.csv`, `data/val.csv`, `data/test.csv` |
| Gold master splits | `data/gold_master/` |
| Active full FT output | `outputs_full_ft/` |
| Completed partial FT | `outputs/final_model/` |
| Archived partial FT baseline | `archive/partial_ft_usc/` |
| Active training log | `logs/full_ft_uzbek.log` |
| Active system log | `logs/full_ft_uzbek_system.log` |
| Full FT status reports | `logs/full_ft_status_reports/` |
| Benchmark reports | `benchmark/reports/` |
| Data quality reports | `reports/gold_quality_report/` |
| Dedup reports | `reports/gold_dedup_report/` |

## Runtime Control

Training should be run in tmux for SSH-safe execution. System monitoring runs in a separate tmux window and writes GPU/CPU/RAM/disk telemetry.

Current active session:

```bash
tmux attach -t whisper_full_ft_uzbek
```
