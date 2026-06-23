# Benchmarking and Evaluation

## Evaluation Goals

The evaluation framework must answer:

- Uzbek WER/CER for each model.
- Normalized WER/CER after canonical Uzbek normalization.
- Hallucination rate.
- Language-confusion rate.
- Inference speed and VRAM.
- Whether a checkpoint beats the archived partial FT baseline.

## Evaluation Scripts

- `benchmark/eval_suite.py`
- `benchmark/language_confusion_benchmark.py`
- `src/evaluate_baseline.py`

Example:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src
python benchmark/eval_suite.py --model-path outputs/final_model --manifest data/test.csv --output reports/eval_partial_ft_test.json
```

## Metrics

Primary:

- WER.
- CER.

Secondary:

- Normalized WER.
- Normalized CER.
- Language-confusion rate.
- Hallucination rate.
- Runtime.
- Samples/sec.
- Peak VRAM.

## Language Confusion

Language confusion is especially important because raw Whisper often outputs Turkish, Kazakh, Cyrillic/Russian-like text, or repeated hallucinations for Uzbek speech.

Use:

```bash
python benchmark/language_confusion_benchmark.py --model-path outputs/final_model --manifest data/test.csv
```

## Inference Benchmarking

Benchmark framework:

- `benchmark/scripts/benchmark_inference.py`
- `benchmark/scripts/run_benchmark.sh`
- `benchmark/scripts/capacity_planner.py`

Engines:

- Hugging Face Transformers.
- faster-whisper.
- CTranslate2.

Existing reports:

- `benchmark/reports/final_capacity_report.md`
- `benchmark/reports/faster_whisper_batch_scaling_a40.md`
- `benchmark/reports/long_form_offline_capacity_report.md`

Known measured inference result:

- Engine: faster-whisper.
- Model: large-v3.
- Hardware: A40.
- Precision: FP16.
- Batch-scaling smoke found batch size 2 best on that workload.
- Best smoke speed was about `10x` real time for batch size 2.

## Benchmark Caveats

- Smoke benchmarks are not production capacity estimates.
- Long-form offline benchmarking is more representative for enterprise transcription.
- Streaming capacity is a separate mode and must not be inferred from offline batch throughput.
- Costs in `benchmark/configs/hardware_costs.yaml` are placeholders and must be updated before financial planning.
