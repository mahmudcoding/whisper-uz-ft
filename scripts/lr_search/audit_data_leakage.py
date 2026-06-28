#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES_WITH_RELIABLE_SPEAKERS = {"usc", "common_voice_uz"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(row: dict[str, str]) -> str:
    return " ".join(row["text"].lower().split())


def audit_subset(path: Path) -> dict:
    manifests = {split: path / f"{split}.csv" for split in ("train", "val", "test")}
    rows = {split: read_manifest(manifest) for split, manifest in manifests.items()}
    errors: list[str] = []
    warnings: list[str] = []

    path_sets = {split: {row["audio_path"] for row in values} for split, values in rows.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = path_sets[left] & path_sets[right]
        if overlap:
            errors.append(f"{len(overlap)} exact audio paths overlap between {left} and {right}")

    for source in SOURCES_WITH_RELIABLE_SPEAKERS:
        speaker_sets = {
            split: {
                row["speaker_id"]
                for row in values
                if row["source_metadata"] == source and row["speaker_id"]
            }
            for split, values in rows.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = speaker_sets[left] & speaker_sets[right]
            if overlap:
                errors.append(f"{source}: {len(overlap)} speakers overlap between {left} and {right}")

    text_sets = {split: {normalized_text(row) for row in values} for split, values in rows.items()}
    text_collisions = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        count = len(text_sets[left] & text_sets[right])
        text_collisions[f"{left}_{right}"] = count
        if count:
            warnings.append(
                f"{count} normalized transcript strings occur in both {left} and {right}; "
                "audio and reliable speakers remain disjoint"
            )

    return {
        "root": str(path.resolve()),
        "manifests": {
            split: {
                "path": str(manifest.resolve()),
                "sha256": file_sha256(manifest),
                "rows": len(rows[split]),
            }
            for split, manifest in manifests.items()
        },
        "exact_path_overlap": {
            f"{left}_{right}": len(path_sets[left] & path_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "transcript_collisions": text_collisions,
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }


def load_resolved_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = config.pop("base_config", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = load_resolved_config(base_path.resolve())
        merged.update(config)
        return merged
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LR-search split integrity and test isolation.")
    parser.add_argument("--output", type=Path, default=Path("reports/lr_search/data_leakage_audit.md"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/lr_search/data_leakage_audit.json"))
    args = parser.parse_args()

    subsets = [
        audit_subset(PROJECT_ROOT / "data/lr_search/coarse_10h"),
        audit_subset(PROJECT_ROOT / "data/lr_search/main_30h"),
    ]
    config_errors = []
    for path in sorted((PROJECT_ROOT / "configs/lr_search").rglob("*.yaml")):
        if path.name.startswith("base_"):
            continue
        config = load_resolved_config(path)
        if config.get("evaluate_test_after_training") is not False:
            config_errors.append(f"{path}: evaluate_test_after_training must be false")
        if config.get("load_test_split") is not False:
            config_errors.append(f"{path}: load_test_split must be false")

    source = (PROJECT_ROOT / "src/train.py").read_text(encoding="utf-8")
    code_checks = {
        "trainer_train_dataset_is_train": 'train_dataset": dataset["train"]' in source,
        "trainer_eval_dataset_is_validation": 'eval_dataset": dataset["validation"]' in source,
        "test_loading_is_configurable": "include_test=bool(cfg.get(\"load_test_split\", True))" in source,
        "test_evaluation_is_guarded": 'if bool(cfg.get("evaluate_test_after_training", True))' in source,
    }
    errors = config_errors + [f"code check failed: {key}" for key, ok in code_checks.items() if not ok]
    errors.extend(error for subset in subsets for error in subset["errors"])
    payload = {
        "status": "pass" if not errors else "fail",
        "subsets": subsets,
        "code_checks": code_checks,
        "config_errors": config_errors,
        "errors": errors,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LR Search Data Leakage Audit",
        "",
        f"- Overall status: **{payload['status'].upper()}**",
        "- Training source: `train.csv` only.",
        "- Checkpoint/model selection source: `val.csv` only.",
        "- Test policy: `test.csv` is neither loaded nor evaluated during LR search.",
        "",
        "## Exact Paths and Hash Guards",
        "",
    ]
    for subset in subsets:
        lines.extend([f"### `{subset['root']}`", ""])
        for split in ("train", "val", "test"):
            item = subset["manifests"][split]
            lines.append(
                f"- {split}: `{item['path']}`; rows `{item['rows']}`; SHA-256 `{item['sha256']}`"
            )
        lines.extend(
            [
                f"- Exact path overlap: `{subset['exact_path_overlap']}`",
                f"- Transcript collisions: `{subset['transcript_collisions']}`",
                f"- Status: **{subset['status'].upper()}**",
                "",
            ]
        )
        for warning in subset["warnings"]:
            lines.append(f"- Warning: {warning}")
        if subset["warnings"]:
            lines.append("")
    lines.extend(
        [
            "## Pipeline Verification",
            "",
            f"- Code checks: `{code_checks}`",
            f"- Config errors: `{config_errors}`",
            "",
            "## Fix Applied",
            "",
            "Before this audit, LR-search runs disabled final test evaluation but still loaded and",
            "feature-preprocessed `test.csv`. The training loader now supports `load_test_split: false`,",
            "all LR-search configs enforce it, the runner rejects any config that loads/evaluates test,",
            "and the comparison tool rejects metrics containing test results.",
            "",
            "Repeated transcript text across splits is not treated as leakage when audio paths and",
            "reliable speaker identities are disjoint: read-speech corpora legitimately contain common",
            "short phrases. Test manifest hashes above are the immutable guards for the search.",
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
