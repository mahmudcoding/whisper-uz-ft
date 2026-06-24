# Evaluation and Inference Benchmarking

**Document role:** Quality metrics, evaluation integrity, benchmark methodology,
measured performance, and capacity-planning caveats.

## Evaluation Principles

1. Use identical manifests and normalization when comparing models.
2. Force Uzbek decoding.
3. Select hyperparameters on validation only.
4. Evaluate locked test data once per promoted configuration.
5. Report dataset, split, normalization, beam size, and decoding policy with every
   metric.
6. Do not compare metrics from different proxy/test sets as if they were equivalent.
7. Treat small WER/CER deltas as uncertain unless confirmed on a larger set.

## Quality Metrics

| Metric | Meaning |
|---|---|
| WER | Word substitutions, deletions, and insertions divided by reference words |
| CER | Character-level error ratio |
| Normalized WER/CER | Metrics after canonical Uzbek normalization |
| Hallucination rate | Fraction triggering repeated-text or extreme-length heuristics |
| Language-confusion rate | Fraction containing Turkish/Kazakh-script indicators |
| Eval loss | Teacher-forced validation loss; supporting, not primary, evidence |

WER/CER are stored as ratios. Values can exceed 1.0 when insertions are severe.

## Evaluation Implementations

- `src/train.py`: validation and optional final test evaluation.
- `src/evaluate_baseline.py`: raw-model baseline evaluation.
- `benchmark/eval_suite.py`: reproducible model comparison.
- `benchmark/language_confusion_benchmark.py`: confusion-focused evaluation.

The current hallucination/confusion indicators are useful guardrails, not complete
linguistic classifiers. Review sample predictions for decision-critical comparisons.

## Baseline Results

| Model | Evaluation | WER | CER |
|---|---|---:|---:|
| Raw Whisper large-v3 | USC mini test | 1.052247 | 0.459004 |
| Mini FT | mini test | 0.496067 | 0.109443 |
| Partial FT USC | protected USC test | **0.200526** | **0.052908** |
| Full FT USC | protected USC test | 0.222152 | 0.056583 |

The protected partial FT is the current quality baseline.

## Inference Benchmark Framework

Location: `benchmark/`.

Engines:

- Hugging Face Transformers;
- faster-whisper;
- CTranslate2 conversion.

Benchmark modes:

- offline batch transcription;
- streaming simulation;
- long-form offline throughput.

Measured fields include:

- audio duration and end-to-end wall time;
- startup overhead;
- mean, p50, p95, p99 latency;
- throughput and real-time factor;
- GPU utilization, VRAM, power, and temperature;
- CPU and RAM;
- WER/CER where references exist.

Core formula:

```text
RTF = processing wall time / audio duration
speed multiplier = 1 / RTF
```

## Primary Commands

Smoke benchmark:

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

Fine-tuned Transformers checkpoint:

```bash
bash benchmark/scripts/run_benchmark.sh \
  --engine transformers \
  --model-path archive/partial_ft_usc/model \
  --dataset smoke \
  --precision fp16
```

Capacity model:

```bash
python benchmark/scripts/capacity_planner.py
```

Convert to CTranslate2:

```bash
bash benchmark/scripts/convert_to_ctranslate2.sh \
  archive/partial_ft_usc/model \
  benchmark/models/partial-ft-ct2 \
  float16
```

## Measured A40 Inference Results

### Smoke Batch Scaling

Configuration: faster-whisper large-v3, FP16, beam 1.

Best smoke result:

- batch 2;
- RTF `0.0998`;
- speed `10.02x`;
- peak VRAM `4,257 MB`.

This is not the production capacity number because the audio set is too small.

### Long-Form Offline

Configuration: faster-whisper large-v3, FP16, beam 1, 5h USC-derived dataset.

| Batch | End-to-end RTF | Throughput h/h | Avg GPU | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 0.0764 | 13.13 | 90.1% | 5,505 MB |
| 2 | 0.0263 | 38.30 | 86.5% | 4,545 MB |
| 4 | **0.0230** | **43.82** | 84.3% | 5,089 MB |

Best measured offline configuration: batch 4, beam 1.

At 70% utilization and $1.20/A40-hour, the report estimates approximately
`$0.0391/audio-hour`. This excludes broader service costs unless configured.

Real-time-equivalent A40 counts from the measured long-form result:

| Streams | A40 GPUs |
|---:|---:|
| 100 | 4 |
| 1,000 | 33 |
| 10,000 | 327 |
| 100,000 | 3,261 |

These are throughput equivalents, not validated low-latency streaming capacity.

## Capacity-Planning Caveats

- Cloud prices in `benchmark/configs/hardware_costs.yaml` are editable assumptions.
- L4/A100/H100 numbers are projections until measured on those GPUs.
- Beam 5 long-form did not complete and must not be inferred from beam 1.
- ASR-only cost excludes networking, orchestration, storage, observability, and queueing.
- Optional VAD, diarization, punctuation, LLM correction, and summarization add latency
  and cost.
- Rebenchmark the winning fine-tuned checkpoint after conversion.

## Authoritative Reports

- `benchmark/reports/faster_whisper_batch_scaling_a40.md`
- `benchmark/reports/long_form_offline_capacity_report.md`
- `benchmark/reports/final_capacity_report.md`
- `benchmark/reports/capacity_plan.json`

Prefer the long-form report over smoke results for offline enterprise planning.
