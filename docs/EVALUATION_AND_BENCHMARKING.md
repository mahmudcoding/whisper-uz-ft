# Evaluation and Benchmarking

Last cleaned for obsolete documentation sections: `2026-07-01T04:52:10Z`.

## Evaluation Policy

Validation WER is the primary model-selection metric. CER is secondary. Test is locked until explicit final evaluation. LR search and Stage 1 use validation only.

## Benchmarks

Inference benchmarking framework lives in `benchmark/`. Historical A40 faster-whisper batch scaling found batch size 2 best on the smoke workload, while long-form offline benchmarks exist under `benchmark/reports/`.

## Current Best Completed Model

`outputs_full_gold/best_model/` achieved validation WER 14.50%, CER 3.67% at step 5000 on the Gold validation split.
