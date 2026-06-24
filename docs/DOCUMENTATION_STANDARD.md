# Documentation Standard

**Document role:** Define ownership, structure, update workflow, and validation for all
project documentation.

## Principles

Documentation is a production asset. It must be:

- factually grounded in current artifacts;
- explicit about timestamp and evidence;
- organized by reader intent;
- nonredundant;
- sufficient for recovery and handoff;
- usable by humans and coding agents;
- updated with meaningful project changes.

## Authoritative Set

| Document | Owns |
|---|---|
| `README.md` | documentation index and authority model |
| `PROJECT_CHARTER.md` | mission, scope, constraints, success |
| `STATUS.md` | live jobs, best model, current data, immediate sequence |
| `ARCHITECTURE.md` | stable components and artifact contracts |
| `ENVIRONMENT_SETUP.md` | host/software installation and validation |
| `DATA_GOVERNANCE.md` | dataset inventory, schemas, normalization, dedup, splits |
| `TRAINING_AND_SEARCH.md` | training behavior and LR-search protocol |
| `EVALUATION_AND_BENCHMARKING.md` | metrics, evaluation, inference, capacity |
| `OPERATIONS_RUNBOOK.md` | launch, monitor, resume, stop, incidents |
| `EXPERIMENT_LEDGER.md` | chronological experiment evidence |
| `MODEL_REGISTRY.md` | model identity, provenance, metrics, retention |
| `DECISION_LOG.md` | consequential choices and reversal conditions |
| `FAILURE_LOG.md` | failures, causes, resolutions, prevention |
| `ROADMAP.md` | prioritized unfinished work |
| `DISASTER_RECOVERY.md` | full rebuild and recovery |
| `AGENT_BRIEF.md` | compressed context for new AI sessions |
| `state.json` | machine-readable current state |

Root `AGENTS.md` owns contributor/Codex behavior. Root `README.md` is the external entry
point. Component READMEs may contain only component-specific quick starts.

## Source of Truth

When facts disagree:

1. inspect current process/config/filesystem/checkpoints/metrics;
2. update `STATUS.md` and `state.json`;
3. update stable manuals if the underlying contract changed;
4. append historical ledgers;
5. preserve old material in `archive/`.

Do not copy a changing status fact into multiple manuals.

## Update Matrix

| Change | Required documentation |
|---|---|
| Job launch/stop/phase transition | `STATUS.md`, `state.json` |
| Completed experiment | `EXPERIMENT_LEDGER.md`, `STATUS.md`, `state.json` |
| Promoted/decision-relevant model | `MODEL_REGISTRY.md` |
| Strategy/config policy change | `DECISION_LOG.md`, relevant manual |
| Failure or operational incident | `FAILURE_LOG.md`, `STATUS.md` if active |
| Dataset acquisition/rebuild | `DATA_GOVERNANCE.md`, `STATUS.md`, `state.json` |
| Environment dependency change | `ENVIRONMENT_SETUP.md`, lock file |
| Training code/contract change | `TRAINING_AND_SEARCH.md`, `ARCHITECTURE.md` if needed |
| Benchmark methodology/result | `EVALUATION_AND_BENCHMARKING.md` and generated report |
| Priority change | `ROADMAP.md` |
| Agent/contributor workflow | root `AGENTS.md` |

## Writing Standard

- Use descriptive headings and short paragraphs.
- Put stable facts in manuals and volatile facts in `STATUS.md`.
- Define units and distinguish ratios from percentages.
- Name exact files, configs, manifests, and reports.
- Separate measured facts from hypotheses and recommendations.
- State limitations and residual risks.
- Include commands that are safe and tested.
- Avoid phrases such as "current" without a timestamp in historical records.
- Do not describe a plan as completed work.
- Do not claim a dataset/model exists without an artifact path.

## New Document Gate

Create a new top-level doc only if:

1. no existing document owns the information;
2. it has a distinct reader task;
3. it will be maintained;
4. it is added to `docs/README.md` and the validator.

Generated, experiment-specific detail belongs under `reports/`, not `docs/`.

## Archival

Before a major documentation rewrite:

1. create a timestamped snapshot under `archive/`;
2. preserve root/component READMEs and `AGENTS.md`;
3. verify the archive contents;
4. replace the live docs;
5. record the rewrite in `FAILURE_LOG.md`.

Historical snapshots are read-only evidence, not current instructions.

## Validation

Run:

```bash
source .venv/bin/activate
python scripts/update_docs.py --check
```

The checker must validate:

- required docs and minimum content;
- no unexpected top-level Markdown;
- JSON schema/required state keys;
- internal Markdown links;
- critical model/data/config values;
- protected baseline references;
- test-isolation policy;
- documentation index coverage;
- stale legacy filenames.

Also run:

```bash
git diff --check
python -m py_compile scripts/update_docs.py
```

## Review Checklist

Before declaring documentation complete:

- Can a new engineer identify the best model and active job in under five minutes?
- Can they reproduce the environment?
- Can they distinguish Gold governance schema from training schema?
- Can they launch or resume without duplicating jobs?
- Can they explain why full FT is not the default?
- Can they prove test data is isolated during search?
- Can they find every decision-relevant model?
- Can they rebuild after losing the server?
- Do all commands and paths exist?
- Does `state.json` match `STATUS.md`?

If any answer is no, the documentation is incomplete.
