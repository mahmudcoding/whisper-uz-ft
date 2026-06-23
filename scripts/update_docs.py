from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED_DOCS = [
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
    "DOCUMENTATION_POLICY.md",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_docs() -> list[str]:
    problems: list[str] = []
    for name in REQUIRED_DOCS:
        path = DOCS / name
        if not path.exists():
            problems.append(f"missing required doc: docs/{name}")
        elif path.stat().st_size < 500:
            problems.append(f"doc is suspiciously short: docs/{name}")
    return problems


def check_no_unstructured_top_level_docs() -> list[str]:
    allowed = set(REQUIRED_DOCS)
    allowed.add("CURRENT_STATE.json")
    problems = []
    for path in DOCS.glob("*.md"):
        if path.name not in allowed:
            problems.append(f"unexpected top-level doc; move into numbered set or docs/archive/: docs/{path.name}")
    return problems


def check_config_consistency() -> list[str]:
    problems: list[str] = []
    cfg_path = ROOT / "configs" / "full_ft_uzbek.yaml"
    if not cfg_path.exists():
        return ["missing configs/full_ft_uzbek.yaml"]
    cfg = yaml.safe_load(read(cfg_path))
    current = read(DOCS / "01_CURRENT_STATE.md") if (DOCS / "01_CURRENT_STATE.md").exists() else ""
    training = read(DOCS / "07_TRAINING_PIPELINE.md") if (DOCS / "07_TRAINING_PIPELINE.md").exists() else ""
    agent = read(DOCS / "15_AI_AGENT_CONTEXT.md") if (DOCS / "15_AI_AGENT_CONTEXT.md").exists() else ""

    if cfg.get("epochs") == 1:
        for name, text in {
            "01_CURRENT_STATE.md": current,
            "07_TRAINING_PIPELINE.md": training,
            "15_AI_AGENT_CONTEXT.md": agent,
        }.items():
            if not re.search(r"epochs?: `?1`?", text, re.IGNORECASE):
                problems.append(f"{name} does not clearly document epochs=1")
    else:
        problems.append(f"configs/full_ft_uzbek.yaml epochs is {cfg.get('epochs')}, expected 1 unless user changed plan")

    expected_pairs = {
        "encoder_learning_rate": "2e-6",
        "decoder_learning_rate": "8e-6",
        "max_grad_norm": "1.0",
    }
    combined = "\n".join([current, training, agent])
    for key, expected in expected_pairs.items():
        value = cfg.get(key)
        if str(value) not in {expected, expected.replace("e-6", ".0e-6")} and float(value) != float(expected):
            problems.append(f"{key} config value {value} does not match expected {expected}")
        if expected not in combined and expected.replace("e-6", ".0e-6") not in combined:
            problems.append(f"docs do not mention expected {key}={expected}")
    return problems


def check_current_state() -> list[str]:
    path = DOCS / "01_CURRENT_STATE.md"
    if not path.exists():
        return ["missing 01_CURRENT_STATE.md"]
    text = read(path)
    required_phrases = [
        "whisper_full_ft_uzbek",
        "scripts/guard_one_epoch_resume.sh",
        "partial_ft_usc_baseline",
        "data/gold_master",
        "207.12h",
    ]
    return [f"01_CURRENT_STATE.md missing phrase: {phrase}" for phrase in required_phrases if phrase not in text]


def run_check() -> int:
    problems = []
    problems.extend(check_required_docs())
    problems.extend(check_no_unstructured_top_level_docs())
    problems.extend(check_config_consistency())
    problems.extend(check_current_state())

    if problems:
        print("DOCUMENTATION CHECK FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("DOCUMENTATION CHECK PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate authoritative project documentation.")
    parser.add_argument("--check", action="store_true", help="Run documentation consistency checks.")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(run_check())
    parser.print_help()


if __name__ == "__main__":
    main()
