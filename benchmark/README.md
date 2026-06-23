# Whisper Inference Benchmarking

This directory contains the production benchmark and capacity planning framework for Whisper inference.

## Main Commands

Run a measured smoke benchmark and regenerate the capacity report:

```bash
bash benchmark/scripts/run_benchmark.sh \
  --engine faster-whisper \
  --model-path large-v3 \
  --dataset smoke \
  --precision fp16 \
  --batch-size 1 \
  --beam-size 1 \
  --mode offline
```

Run a local fine-tuned checkpoint with HuggingFace Transformers:

```bash
bash benchmark/scripts/run_benchmark.sh \
  --engine transformers \
  --model-path outputs/mini/final_model \
  --dataset smoke \
  --precision fp16
```

Run the built-in suite for the selected engine/model/dataset:

```bash
bash benchmark/scripts/run_benchmark.sh --full-suite --engine faster-whisper --model-path large-v3 --dataset smoke
```

Convert a HuggingFace checkpoint to CTranslate2/faster-whisper format:

```bash
bash benchmark/scripts/convert_to_ctranslate2.sh outputs/mini/final_model benchmark/models/mini-final-ct2 float16
```

Regenerate capacity planning from all benchmark JSON files:

```bash
python benchmark/scripts/capacity_planner.py
```

## Outputs

- Benchmark JSON results: `benchmark/results/*.json`
- Timestamped GPU/CPU/RAM telemetry: `benchmark/results/gpu_metrics.csv`
- Capacity plan JSON: `benchmark/reports/capacity_plan.json`
- Production report: `benchmark/reports/final_capacity_report.md`
- Editable hardware and pipeline costs: `benchmark/configs/hardware_costs.yaml`
