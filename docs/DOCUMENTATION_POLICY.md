# Documentation Policy

Documentation is a production asset for this project.

## Authority

The authoritative docs are:

- `00_PROJECT_OVERVIEW.md`
- `01_CURRENT_STATE.md`
- `02_ARCHITECTURE.md`
- `03_ENVIRONMENT_SETUP.md`
- `04_DATASETS.md`
- `05_DATA_PIPELINE.md`
- `06_TEXT_NORMALIZATION.md`
- `07_TRAINING_PIPELINE.md`
- `08_EXPERIMENT_HISTORY.md`
- `09_BENCHMARKING.md`
- `10_MODEL_REGISTRY.md`
- `11_DECISIONS_AND_RATIONALE.md`
- `12_FAILURES_AND_LESSONS.md`
- `13_TODO_AND_NEXT_STEPS.md`
- `14_RECOVERY_GUIDE.md`
- `15_AI_AGENT_CONTEXT.md`

Archived docs under `docs/archive/` are historical reference only.

## Required Updates

Every meaningful code/config/data/training change must update docs in the same work session.

Rules:

1. Every experiment must be logged in `08_EXPERIMENT_HISTORY.md`.
2. Every model/checkpoint used for decisions must be entered in `10_MODEL_REGISTRY.md`.
3. Every strategy change must be explained in `11_DECISIONS_AND_RATIONALE.md`.
4. Every failure or wrong assumption must be captured in `12_FAILURES_AND_LESSONS.md`.
5. Current running state must be reflected in `01_CURRENT_STATE.md`.
6. New datasets or corpus changes must update `04_DATASETS.md` and `05_DATA_PIPELINE.md`.
7. New setup/dependency requirements must update `03_ENVIRONMENT_SETUP.md`.
8. AI agents must read `15_AI_AGENT_CONTEXT.md` before modifying code.

## No Redundancy Rule

Do not create new topical docs unless they are clearly needed.

If a topic overlaps with an existing numbered doc:

- Update the numbered doc.
- Do not add a duplicate standalone doc.

Historical detailed reports can be stored under:

```bash
docs/archive/
```

## Staleness Checks

Run:

```bash
python scripts/update_docs.py --check
```

The checker validates:

- Required docs exist.
- Archived docs are not treated as top-level docs.
- Current config values mentioned in docs match critical config values.
- Current-state docs mention active training state and one-epoch guard.

## AI Agent Workflow

Before work:

1. Read `15_AI_AGENT_CONTEXT.md`.
2. Read `01_CURRENT_STATE.md`.
3. Read `11_DECISIONS_AND_RATIONALE.md`.
4. Inspect relevant source files.

After work:

1. Update affected numbered docs.
2. Run documentation check.
3. Include changed docs in final summary.

## Documentation Quality Bar

Docs must be:

- Precise.
- Current.
- Unambiguous.
- Reconstructable from zero.
- Useful for humans and AI agents.
- Explicit about failures and blockers.
