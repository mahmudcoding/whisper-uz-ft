# Architecture

Generated: 2026-06-23 UTC

## Training Pipeline

1. CSV manifests in `data/` define `audio_path`, `text`, and `duration`.
2. `src/data_loader.py` loads audio and prepares Whisper features/labels.
3. `src/model.py` loads `openai/whisper-large-v3` and applies freeze policy.
4. `src/train.py` configures the Hugging Face trainer, safety callbacks, checkpoint validation, resume behavior, and final test evaluation.
5. `scripts/start_full_training.sh` launches training inside tmux/safe background execution.
6. `scripts/system_monitor.sh` logs GPU/CPU/RAM/disk telemetry.
7. Artifacts are written to `outputs/`, logs to `logs/`, metrics to TensorBoard JSONL and JSON files.

## Evaluation Pipeline

1. `benchmark/eval_suite.py` loads a model and fixed manifest.
2. It generates transcriptions with the requested precision and batch size.
3. It computes raw WER/CER and normalized WER/CER.
4. It records runtime and peak GPU memory.

## Inference Benchmark Pipeline

1. `benchmark/scripts/create_benchmark_datasets.py` and related scripts create manifests.
2. `benchmark/scripts/benchmark_inference.py` runs engines such as Transformers and faster-whisper.
3. `benchmark/scripts/capacity_planner.py` converts measured throughput into GPU/server/cost estimates.
4. Reports are written to `benchmark/reports/`.

## Data Quality Pipeline

1. `src/text_normalization/uz_normalizer.py` normalizes Uzbek Latin/Cyrillic/mixed-script text.
2. `src/filtering/scoring.py` assigns sample quality scores.
3. `src/filtering/filter_dataset.py` writes scored manifests and suspicious samples.
4. Future teacher-ASR scoring should add transcript/audio similarity.

## Uzbek-Only Addendum

Generated: 2026-06-23 UTC

Full fine-tuning mode:

- Config: `configs/full_ft_uzbek.yaml`
- `train_last_encoder_blocks: all`
- all model parameters trainable
- BF16 enabled
- output in `outputs_full_ft`

Decoding policy:

- all supported inference/evaluation paths force `language=uz`, `task=transcribe`
- automatic language detection is disallowed

Language confusion:

- `benchmark/language_confusion_benchmark.py` wraps the eval suite and flags Turkish/Kazakh/Russian/English leakage.
