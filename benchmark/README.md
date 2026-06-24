# Whisper Evaluation, Benchmarking, and Capacity Planning

This directory contains model-quality evaluation, language-confusion analysis,
inference performance measurement, GPU telemetry, and hardware/cost planning.

The authoritative methodology and measured results are documented in
[`docs/EVALUATION_AND_BENCHMARKING.md`](../docs/EVALUATION_AND_BENCHMARKING.md).

## Layout

| Path | Purpose |
|---|---|
| `eval_suite.py` | compare ASR models on a fixed manifest |
| `language_confusion_benchmark.py` | target Uzbek language/script confusion |
| `scripts/benchmark_inference.py` | measured inference runner |
| `scripts/run_benchmark.sh` | benchmark orchestration |
| `scripts/capacity_planner.py` | GPU/server/cost projection |
| `scripts/convert_to_ctranslate2.sh` | Hugging Face to CTranslate2 conversion |
| `scripts/create_benchmark_datasets.py` | benchmark manifest generation |
| `scripts/create_long_form_offline_dataset.py` | 5h long-form workload |
| `configs/` | suite and hardware-cost assumptions |
| `datasets/` | benchmark manifests |
| `results/` | measured JSON and telemetry |
| `reports/` | generated Markdown, CSV, and capacity outputs |

## Quick Start

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src

bash benchmark/scripts/run_benchmark.sh \
  --engine faster-whisper \
  --model-path large-v3 \
  --dataset smoke \
  --precision fp16 \
  --batch-size 1 \
  --beam-size 1 \
  --mode offline
```

Benchmark a fine-tuned Transformers checkpoint:

```bash
bash benchmark/scripts/run_benchmark.sh \
  --engine transformers \
  --model-path archive/partial_ft_usc/model \
  --dataset smoke \
  --precision fp16
```

Convert a checkpoint:

```bash
bash benchmark/scripts/convert_to_ctranslate2.sh \
  archive/partial_ft_usc/model \
  benchmark/models/partial-ft-ct2 \
  float16
```

Regenerate capacity outputs:

```bash
python benchmark/scripts/capacity_planner.py
```

## Interpreting Results

```text
RTF = wall time / audio duration
speed multiplier = 1 / RTF
```

Use smoke runs only to validate functionality. For offline planning, prefer
`reports/long_form_offline_capacity_report.md`, based on 5h audio.

Current best measured A40 long-form result:

- faster-whisper large-v3;
- FP16;
- beam 1;
- batch 4;
- RTF 0.0230;
- 43.82 audio-hours/hour/GPU;
- peak VRAM 5,089 MB.

Do not interpret these throughput-equivalent numbers as measured low-latency streaming
capacity. L4/A100/H100 results remain modeled until measured on those devices.

## Integrity Rules

- Report model path, engine, precision, beam, batch, dataset, and mode.
- Keep quality evaluation normalization consistent.
- Preserve raw result JSON and telemetry.
- Do not use benchmark test data to tune training.
- Rebenchmark the promoted fine-tuned checkpoint after CTranslate2 conversion.
- Treat costs in `configs/hardware_costs.yaml` as editable assumptions.
