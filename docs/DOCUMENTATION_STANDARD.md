# Documentation Standard

Last rebuilt from repository reality: `2026-07-01T04:52:10Z`.

For current live state, read `STATUS.md` first. For full memory transfer, read `../PROJECT_CONTEXT_EXPORT.txt`.

## Policy

Every meaningful code, config, data, training, evaluation, cleanup, or decision change must update docs in the same work session.

Required updates:

- Runtime changes: `STATUS.md` and `docs/state.json`
- Dataset changes: `DATA_GOVERNANCE.md`
- Training/search changes: `TRAINING_AND_SEARCH.md` and `EXPERIMENT_LEDGER.md`
- Decisions: `DECISION_LOG.md`
- Failures: `FAILURE_LOG.md`
- Model promotion: `MODEL_REGISTRY.md`
- Cross-session memory: `../PROJECT_CONTEXT_EXPORT.txt`

Live artifacts override docs when they disagree, but the disagreement must be resolved promptly.
