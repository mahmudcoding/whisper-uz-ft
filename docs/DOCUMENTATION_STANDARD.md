# Documentation Standard

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

## Policy

Every meaningful code, config, data, training, evaluation, cleanup, or decision change must update docs in the same work session.

Required updates:

- Stable project status or artifact changes: `STATUS.md` and `docs/state.json`
- Dataset changes: `DATA_GOVERNANCE.md`
- Training/search changes: `TRAINING_AND_SEARCH.md` and `EXPERIMENT_LEDGER.md`
- Decisions: `DECISION_LOG.md`
- Failures: `FAILURE_LOG.md`
- Model promotion: `MODEL_REGISTRY.md`

Code, configs, manifests, metrics, and reports are authoritative when they disagree with docs, but the disagreement must be resolved promptly.
