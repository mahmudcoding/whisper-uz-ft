# Project Documentation

This directory is the authoritative technical and operational record for the Uzbek
Whisper ASR project. It is organized by reader intent rather than chronology.

**Last structural rebuild:** 2026-06-24 UTC  
**Project root:** `/home/mahmud/whisper-uz-ft`  
**Raw dataset root:** `/home/mahmud/datasets`

## Start Here

| Need | Read |
|---|---|
| Understand the mission, scope, and success criteria | [PROJECT_CHARTER.md](PROJECT_CHARTER.md) |
| See what is running and what matters now | [STATUS.md](STATUS.md) |
| Understand components and artifact flow | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Install or repair the environment | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) |
| Work with datasets and split governance | [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md) |
| Train models or continue LR search | [TRAINING_AND_SEARCH.md](TRAINING_AND_SEARCH.md) |
| Evaluate quality or benchmark inference | [EVALUATION_AND_BENCHMARKING.md](EVALUATION_AND_BENCHMARKING.md) |
| Launch, monitor, resume, or troubleshoot jobs | [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) |
| Review completed and active experiments | [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) |
| Locate a model or checkpoint | [MODEL_REGISTRY.md](MODEL_REGISTRY.md) |
| Understand why the system is designed this way | [DECISION_LOG.md](DECISION_LOG.md) |
| Avoid known mistakes | [FAILURE_LOG.md](FAILURE_LOG.md) |
| See prioritized future work | [ROADMAP.md](ROADMAP.md) |
| Recover from repository or server loss | [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) |
| Inject context into a new coding-agent session | [AGENT_BRIEF.md](AGENT_BRIEF.md) |
| Maintain these docs correctly | [DOCUMENTATION_STANDARD.md](DOCUMENTATION_STANDARD.md) |

Machine-readable live state is stored in [state.json](state.json).

## Authority Model

Use this order when sources disagree:

1. Current process, filesystem, checkpoint, config, and metric evidence.
2. `STATUS.md` and `state.json`.
3. Stable reference manuals in this directory.
4. Generated reports under `reports/` and `benchmark/reports/`.
5. Historical snapshots under `archive/`.

Generated reports are evidence, not substitutes for the manuals. Historical
documentation is preserved at:

```text
archive/documentation_snapshot_20260624T084325Z/
```

## Core Invariants

- Optimize Uzbek WER/CER; multilingual retention is not a goal.
- Force `language="uz"` and `task="transcribe"`.
- Never modify `archive/partial_ft_usc/`.
- Never use test data for training or hyperparameter selection.
- Do not train unfiltered Silver or Bronze data.
- Do not start duplicate training controllers or overwrite checkpoint directories.
- Update documentation in the same work session as meaningful project changes.

## Documentation Health Check

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
python scripts/update_docs.py --check
```

The checker validates required files, internal links, critical metrics/config values,
machine-readable state, and documentation policy.
