# Project State

Generated: 2026-06-23 UTC

## Summary

This project fine-tunes `openai/whisper-large-v3` for Uzbek ASR using the ISSAI Uzbek Speech Corpus. The current full one-epoch partial fine-tuning run completed successfully on the NVIDIA A40 48 GB server and produced `outputs/final_model`.

Current best full-run test metrics:

| Model | Split | WER | CER |
| --- | --- | ---: | ---: |
| Raw Whisper large-v3 | test | 1.0522 | 0.4590 |
| Mini fine-tuned run | mini/test | 0.4961 | 0.1094 |
| Full partial fine-tuned run | test | 0.2005 | 0.0529 |

## Environment

- Host root: `/home/mahmud/whisper-uz-ft`
- OS kernel: Linux 6.8.0-101-generic
- Python: 3.12.3
- GPU: NVIDIA A40 48 GB
- Driver: 580.159.03
- CUDA runtime reported by `nvidia-smi`: 13.0
- PyTorch: 2.5.1+cu121
- CUDA available in PyTorch: yes
- A40 BF16 support in PyTorch: yes

Key package versions:

| Package | Version |
| --- | --- |
| torch | 2.5.1+cu121 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| evaluate | 0.4.6 |
| accelerate | 1.14.0 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.0 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |

## Repository Layout

Important top-level paths:

| Path | Purpose |
| --- | --- |
| `configs/` | Training configs: dry run, mini run, full run |
| `data/` | Train/val/test CSV manifests |
| `src/` | Training, model, data loading, normalization, filtering |
| `scripts/` | Launch and monitoring scripts |
| `benchmark/` | Inference benchmarks, capacity planner, eval suite |
| `outputs/` | Checkpoints, final model, metrics |
| `logs/` | Training logs, system monitor logs, validation reports |
| `reports/` | Generated audit and feasibility CSV/JSON outputs |
| `docs/` | Engineering documentation |

Notable source modules:

- `src/train.py`: main training entry point with resume, checkpoint validation, safety callback, and final test evaluation.
- `src/model.py`: Whisper loading and freeze policy.
- `src/data_loader.py`: CSV/audio loading and feature preparation.
- `src/text_normalization/uz_normalizer.py`: production Uzbek normalization.
- `src/filtering/filter_dataset.py`: reusable dataset quality scoring CLI.
- `benchmark/eval_suite.py`: reproducible model evaluation CLI.

## Training Status

Training session `whisper_full_training` completed the full one-epoch run. The Python training process exited after final test evaluation.

The tmux session currently only contains the monitor window. It can be closed manually after inspection:

```bash
tmux kill-session -t whisper_full_training
```

Final artifacts:

| Artifact | Status |
| --- | --- |
| `outputs/final_model/model.safetensors` | present, 5.8 GB |
| `outputs/checkpoint-3114` | present, validated, 14 GB |
| `outputs/test_metrics.json` | present |
| `logs/full_training.log` | present |
| `logs/full_training_system.log` | present |

Checkpoints retained:

| Checkpoint | Size | Notes |
| --- | ---: | --- |
| `outputs/checkpoint-2500` | 14 GB | resumable |
| `outputs/checkpoint-3000` | 14 GB | resumable |
| `outputs/checkpoint-3114` | 14 GB | final epoch checkpoint |

Disk status at audit time:

- Filesystem size: 2.0 TB
- Used: 579 GB
- Available: 1.4 TB
- `outputs/`: 114 GB

## Current Training Config

From `configs/train.yaml`:

| Setting | Value |
| --- | --- |
| Base model | `openai/whisper-large-v3` |
| Epochs | 1 |
| Per-device train batch | 2 |
| Gradient accumulation | 16 |
| Effective batch | 32 |
| Learning rate | 1e-5 |
| Scheduler | cosine |
| Warmup steps | 1000 |
| Weight decay | 0.01 |
| Eval steps | 500 |
| Save steps | 500 |
| Precision | fp16 |
| Gradient checkpointing | true |
| Max grad norm | 1.0 |
| Save total limit | 3 |
| Generation beams | 1 |
| Trainable policy | decoder + last 8 encoder blocks |

Trainable parameters:

- Trainable: 1,063,930,880
- Frozen: 479,559,680
- Frozen encoder blocks: 0-23
- Trainable encoder blocks: 24-31

## Known Risks

1. USC is clean read speech and not diverse enough for enterprise meetings, calls, podcasts, and noisy speech.
   - Impact: model may underperform in real-world domains despite strong USC test metrics.
   - Mitigation: scale data to 500-1500 hours with diverse audio and filtering.

2. Current filtering marks duplicate transcripts as suspicious but does not remove them.
   - Impact: repeated labels can bias training if many are boilerplate.
   - Mitigation: review `bad_samples.csv` and build a removal policy after teacher-ASR scoring.

3. Rubai comparison is based on prior analysis notes, not a locally checked-out Rubai repository.
   - Impact: implementation details may drift from upstream.
   - Mitigation: clone and pin the exact Rubai commit before copying ideas.

## Uzbek-Only Pivot Addendum

Generated: 2026-06-23 UTC

The current objective is Uzbek-only ASR quality. Multilingual preservation is explicitly not a goal. The codebase now supports full fine-tuning and forced Uzbek decoding.

New/updated artifacts:

- `configs/full_ft_uzbek.yaml`
- `benchmark/language_confusion_benchmark.py`
- `logs/full_ft_sanity_report.json`
- `reports/teacher_subset_final_model_20.json`
- `reports/language_confusion_smoke.json`
- `reports/project_tree_audit.txt`

Current training status:

- No `python src/train.py` process is running.
- No `whisper_full_training` tmux session is active.
- Previous training stopped because it completed successfully.

Full fine-tuning status:

- Structurally validated.
- All 1,543,490,560 parameters are trainable with `train_last_encoder_blocks: all`.
- CUDA forward pass succeeded.
- Optimizer-step VRAM has not yet been validated; run a 100-300 step dry run before a full launch.
