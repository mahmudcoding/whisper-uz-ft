# Evaluation Framework

Generated: 2026-06-23 UTC

## Module

Implemented:

- `benchmark/eval_suite.py`

Purpose:

- Evaluate any Whisper-compatible model on a fixed CSV manifest.
- Report raw WER/CER and normalized WER/CER.
- Measure runtime, samples/sec, audio-hours/hour, and RTF.
- Capture peak VRAM when CUDA is available.
- Load audio through `soundfile`/`librosa`, avoiding a hard dependency on system `ffmpeg`.

## Current Baselines

| Model | WER | CER | Notes |
| --- | ---: | ---: | --- |
| Raw Whisper large-v3 | 1.0522 | 0.4590 | User-provided baseline |
| Mini fine-tuned checkpoint | 0.4961 | 0.1094 | Existing mini run |
| Full partial fine-tuned checkpoint | 0.2005 | 0.0529 | `outputs/test_metrics.json` |

## Example Commands

Evaluate current final model:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
PYTHONPATH=src python benchmark/eval_suite.py \
  --model-path outputs/final_model \
  --manifest data/test.csv \
  --output reports/eval_final_model_test.json \
  --batch-size 2 \
  --precision fp16
```

Evaluate raw Whisper large-v3:

```bash
PYTHONPATH=src python benchmark/eval_suite.py \
  --model-path openai/whisper-large-v3 \
  --manifest data/test.csv \
  --output reports/eval_raw_large_v3_test.json \
  --batch-size 2 \
  --precision fp16
```

Public model comparison targets:

- `islomov/rubaistt_v2_medium`
- `Kotib/uzbek_stt_v1`

Use the same test manifest and the same normalizer for all models.

## Smoke Test

Validated command:

```bash
PYTHONPATH=src python benchmark/eval_suite.py \
  --model-path outputs/final_model \
  --manifest data/mini_test.csv \
  --max-samples 1 \
  --batch-size 1 \
  --precision fp16 \
  --output reports/eval_suite_smoke.json
```

Smoke result:

- samples: 1
- audio_seconds: 2.04
- RTF: 0.55
- peak_vram_mib: 4184.8

## Required Next Evaluation

Run a real-domain evaluation set with:

- meetings
- calls
- webinars
- podcasts
- noisy speech
- Latin/Cyrillic mixed-script transcripts
- Russian/English code-switching

USC test WER alone is not sufficient for enterprise quality claims.

## Uzbek-Only Evaluation Addendum

Generated: 2026-06-23 UTC

The evaluation suite now forces Uzbek by restricting `--language` to `uz`. It also loads audio with `soundfile`/`librosa`, so it does not require system `ffmpeg`.

Additional tool:

- `benchmark/language_confusion_benchmark.py`

The framework can evaluate:

- `openai/whisper-large-v3`
- `outputs/final_model`
- `islomov/rubaistt_v2_medium`
- `Kotib/uzbek_stt_v1`

Rubai and Kotib were not downloaded/evaluated in this pass. Run them on the same manifest before making public SOTA claims.
