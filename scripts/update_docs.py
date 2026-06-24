from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED_DOCS = [
    "README.md",
    "PROJECT_CHARTER.md",
    "STATUS.md",
    "ARCHITECTURE.md",
    "ENVIRONMENT_SETUP.md",
    "DATA_GOVERNANCE.md",
    "TRAINING_AND_SEARCH.md",
    "EVALUATION_AND_BENCHMARKING.md",
    "OPERATIONS_RUNBOOK.md",
    "EXPERIMENT_LEDGER.md",
    "MODEL_REGISTRY.md",
    "DECISION_LOG.md",
    "FAILURE_LOG.md",
    "ROADMAP.md",
    "DISASTER_RECOVERY.md",
    "AGENT_BRIEF.md",
    "DOCUMENTATION_STANDARD.md",
]

REQUIRED_ROOT_DOCS = ["README.md", "AGENTS.md"]

LEGACY_DOC_NAMES = {
    "00_PROJECT_OVERVIEW.md",
    "01_CURRENT_STATE.md",
    "02_ARCHITECTURE.md",
    "03_ENVIRONMENT_SETUP.md",
    "04_DATASETS.md",
    "05_DATA_PIPELINE.md",
    "06_TEXT_NORMALIZATION.md",
    "07_TRAINING_PIPELINE.md",
    "08_EXPERIMENT_HISTORY.md",
    "09_BENCHMARKING.md",
    "10_MODEL_REGISTRY.md",
    "11_DECISIONS_AND_RATIONALE.md",
    "12_FAILURES_AND_LESSONS.md",
    "13_TODO_AND_NEXT_STEPS.md",
    "14_RECOVERY_GUIDE.md",
    "15_AI_AGENT_CONTEXT.md",
    "CURRENT_STATE.json",
    "DOCUMENTATION_POLICY.md",
    "LR_SEARCH_PLAN.md",
}

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(read(path))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(read(path)) or {}


def check_required_files() -> list[str]:
    problems: list[str] = []
    for name in REQUIRED_DOCS:
        path = DOCS / name
        if not path.is_file():
            problems.append(f"missing required documentation: docs/{name}")
        elif path.stat().st_size < 800:
            problems.append(f"documentation is suspiciously short: docs/{name}")
    for name in REQUIRED_ROOT_DOCS:
        path = ROOT / name
        if not path.is_file():
            problems.append(f"missing root documentation: {name}")
    state = DOCS / "state.json"
    if not state.is_file():
        problems.append("missing machine-readable state: docs/state.json")
    return problems


def check_top_level_structure() -> list[str]:
    allowed = set(REQUIRED_DOCS)
    problems: list[str] = []
    for path in DOCS.glob("*.md"):
        if path.name not in allowed:
            problems.append(f"unexpected top-level documentation file: docs/{path.name}")
    for name in LEGACY_DOC_NAMES:
        if (DOCS / name).exists():
            problems.append(f"legacy documentation filename remains live: docs/{name}")
    return problems


def resolve_local_link(source: Path, target: str) -> Path | None:
    clean = target.strip().strip("<>")
    if not clean or clean.startswith(("#", "http://", "https://", "mailto:")):
        return None
    clean = clean.split("#", 1)[0]
    if not clean:
        return None
    return (source.parent / clean).resolve()


def check_internal_links() -> list[str]:
    problems: list[str] = []
    sources = [ROOT / "README.md", ROOT / "AGENTS.md", *sorted(DOCS.glob("*.md"))]
    sources.extend(
        [
            ROOT / "benchmark/README.md",
            ROOT / "scripts/download_datasets/README.md",
        ]
    )
    for source in sources:
        if not source.is_file():
            continue
        for target in LINK_RE.findall(read(source)):
            resolved = resolve_local_link(source, target)
            if resolved is not None and not resolved.exists():
                problems.append(
                    f"broken local link in {source.relative_to(ROOT)}: {target}"
                )
    return problems


def check_index_coverage() -> list[str]:
    index = read(DOCS / "README.md")
    problems = []
    for name in REQUIRED_DOCS:
        if name == "README.md":
            continue
        if name not in index:
            problems.append(f"docs/README.md does not link or name {name}")
    if "state.json" not in index:
        problems.append("docs/README.md does not reference state.json")
    return problems


def check_state_schema() -> list[str]:
    path = DOCS / "state.json"
    try:
        state = load_json(path)
    except Exception as exc:
        return [f"invalid docs/state.json: {exc!r}"]

    problems: list[str] = []
    required_top = {
        "schema_version",
        "last_verified_utc",
        "project",
        "hardware",
        "best_model",
        "completed_full_ft",
        "gold_master",
        "lr_search",
        "environment",
        "documentation",
    }
    missing = required_top - set(state)
    if missing:
        problems.append(f"docs/state.json missing keys: {sorted(missing)}")
        return problems
    if state["schema_version"] != 2:
        problems.append(f"unexpected state schema version: {state['schema_version']}")
    try:
        datetime.fromisoformat(state["last_verified_utc"].replace("Z", "+00:00"))
    except Exception:
        problems.append("last_verified_utc is not ISO-8601")
    if state["best_model"].get("protected") is not True:
        problems.append("best model must be marked protected")
    if state["lr_search"].get("test_loaded") is not False:
        problems.append("LR-search state must report test_loaded=false")
    if state["lr_search"].get("test_evaluated") is not False:
        problems.append("LR-search state must report test_evaluated=false")
    return problems


def check_metrics_and_data() -> list[str]:
    state = load_json(DOCS / "state.json")
    problems: list[str] = []

    partial_metrics = load_json(
        ROOT / "archive/partial_ft_usc/metrics/test_metrics.json"
    )
    full_metrics = load_json(ROOT / "outputs_full_ft/test_metrics.json")
    best = state["best_model"]
    full = state["completed_full_ft"]

    for key, actual in {
        "test_wer": partial_metrics["test_wer"],
        "test_cer": partial_metrics["test_cer"],
    }.items():
        if abs(float(best[key]) - float(actual)) > 1e-12:
            problems.append(f"best-model {key} does not match protected metrics")
    for key, actual in {
        "test_wer": full_metrics["test_wer"],
        "test_cer": full_metrics["test_cer"],
    }.items():
        if abs(float(full[key]) - float(actual)) > 1e-12:
            problems.append(f"full-FT {key} does not match metrics file")

    gold_report = load_json(
        ROOT / "reports/gold_quality_report/master_validation.json"
    )
    gold = state["gold_master"]
    expected = {
        "rows": gold_report["total_rows"],
        "hours": gold_report["total_hours"],
        "missing_audio_paths": gold_report["missing_audio_paths"],
        "path_leakage": gold_report["path_leakage_across_splits"],
        "content_hash_leakage": gold_report["content_hash_leakage_across_splits"],
        "known_speaker_leakage": gold_report["known_speaker_leakage_across_splits"],
    }
    for key, actual in expected.items():
        if abs(float(gold[key]) - float(actual)) > 1e-9:
            problems.append(f"Gold state {key} does not match validation report")
    return problems


def check_config_policy() -> list[str]:
    problems: list[str] = []
    full = load_yaml(ROOT / "configs/full_ft_uzbek.yaml")
    if int(full.get("epochs", -1)) != 1:
        problems.append("configs/full_ft_uzbek.yaml must document the completed one-epoch run")
    if float(full.get("encoder_learning_rate", 0)) != 2e-6:
        problems.append("full-FT encoder_learning_rate differs from documented 2e-6")
    if float(full.get("decoder_learning_rate", 0)) != 8e-6:
        problems.append("full-FT decoder_learning_rate differs from documented 8e-6")
    if float(full.get("max_grad_norm", 0)) != 1.0:
        problems.append("full-FT max_grad_norm differs from documented 1.0")

    for path in sorted((ROOT / "configs/lr_search").glob("*.yaml")):
        if path.name.startswith("base_"):
            continue
        config = load_resolved_config(path)
        if config.get("load_test_split") is not False:
            problems.append(f"{path.relative_to(ROOT)} must set load_test_split=false")
        if config.get("evaluate_test_after_training") is not False:
            problems.append(
                f"{path.relative_to(ROOT)} must set evaluate_test_after_training=false"
            )
        if config.get("language") != "uz" or config.get("task") != "transcribe":
            problems.append(f"{path.relative_to(ROOT)} must force Uzbek transcription")
    return problems


def load_resolved_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    base = config.pop("base_config", None)
    if not base:
        return config
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = load_resolved_config(base_path.resolve())
    merged.update(config)
    return merged


def check_content_consistency() -> list[str]:
    status = read(DOCS / "STATUS.md")
    registry = read(DOCS / "MODEL_REGISTRY.md")
    training = read(DOCS / "TRAINING_AND_SEARCH.md")
    agent = read(DOCS / "AGENT_BRIEF.md")
    root_agents = read(ROOT / "AGENTS.md")
    combined = "\n".join([status, registry, training, agent, root_agents])
    required_phrases = [
        "partial_ft_usc_baseline",
        "archive/partial_ft_usc",
        "0.2005258480",
        "0.0529079419",
        "data/gold_master",
        "207.1150",
        "language=\"uz\"",
        "task=\"transcribe\"",
        "load_test_split: false",
        "evaluate_test_after_training: false",
        "whisper_lr_search",
    ]
    problems = []
    for phrase in required_phrases:
        if phrase not in combined:
            problems.append(f"authoritative docs missing critical phrase: {phrase}")
    return problems


def check_archive_snapshot() -> list[str]:
    state = load_json(DOCS / "state.json")
    archive = ROOT / state["documentation"]["archived_snapshot"]
    required = [
        archive / "docs",
        archive / "README.md",
        archive / "AGENTS.md",
        archive / "benchmark/README.md",
        archive / "scripts/download_datasets/README.md",
    ]
    return [
        f"documentation archive snapshot missing: {path.relative_to(ROOT)}"
        for path in required
        if not path.exists()
    ]


def run_check() -> int:
    checks = [
        check_required_files,
        check_top_level_structure,
        check_internal_links,
        check_index_coverage,
        check_state_schema,
        check_metrics_and_data,
        check_config_policy,
        check_content_consistency,
        check_archive_snapshot,
    ]
    problems: list[str] = []
    for check in checks:
        try:
            problems.extend(check())
        except Exception as exc:
            problems.append(f"{check.__name__} raised {exc!r}")

    if problems:
        print("DOCUMENTATION CHECK FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("DOCUMENTATION CHECK PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate project documentation structure and factual consistency."
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(run_check())
    parser.print_help()


if __name__ == "__main__":
    main()
