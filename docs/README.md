# Project Documentation

Last synchronized with live repository state: 2026-06-28T16:56:49Z

This directory is the authoritative memory for the Uzbek Whisper ASR project. Live
processes, configs, manifests, metrics, and logs remain the ground truth; docs must be
updated when they diverge.

## Read Order

| Need | File |
|---|---|
| Quick AI handoff | `AGENT_BRIEF.md` |
| Current live state | `STATUS.md` |
| Training and LR-search details | `TRAINING_AND_SEARCH.md` |
| Dataset governance | `DATA_GOVERNANCE.md` |
| Model registry | `MODEL_REGISTRY.md` |
| Experiment history | `EXPERIMENT_LEDGER.md` |
| Decisions | `DECISION_LOG.md` |
| Failures and lessons | `FAILURE_LOG.md` |
| Project charter | `PROJECT_CHARTER.md` |
| Architecture | `ARCHITECTURE.md` |
| Environment setup | `ENVIRONMENT_SETUP.md` |
| Evaluation/benchmarks | `EVALUATION_AND_BENCHMARKING.md` |
| Operations | `OPERATIONS_RUNBOOK.md` |
| Roadmap | `ROADMAP.md` |
| Recovery | `DISASTER_RECOVERY.md` |
| Documentation policy | `DOCUMENTATION_STANDARD.md` |

Machine-readable state is `state.json`. Full cross-session memory export is
`../PROJECT_CONTEXT_EXPORT.txt`.

## Authority Order

1. Current runtime, filesystem, configs, manifests, metrics, and logs.
2. `STATUS.md`, `AGENT_BRIEF.md`, and `state.json`.
3. Decision, failure, training, data, model, and experiment docs.
4. Generated reports under `reports/` and `benchmark/reports/`.
5. Git history.

## Core Invariants

- Optimize Uzbek WER/CER only.
- Force `language="uz"` and `task="transcribe"`.
- Do not modify `models/partial_ft_usc_baseline/`.
- Do not use test data for model selection.
- Do not train unfiltered Silver/Bronze.
- Do not run duplicate long training jobs.
- Keep docs synchronized with meaningful changes.
