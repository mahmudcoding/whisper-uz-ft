#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TARGETS = {"usc": 0.50, "common_voice_uz": 0.40, "fleurs_uz": 0.10}
UNRELIABLE_SPEAKER_SOURCES = {"fleurs_uz"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"audio_path", "text", "duration", "speaker_id", "split", "source_metadata"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return rows


def validate_subset(path: Path, expected_hours: float, tolerance_hours: float) -> dict:
    splits = {name: read(path / f"{name}.csv") for name in ("train", "val", "test")}
    errors: list[str] = []
    warnings: list[str] = []

    paths = {name: {row["audio_path"] for row in rows} for name, rows in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = paths[left] & paths[right]
        if overlap:
            errors.append(f"{len(overlap)} audio paths overlap between {left} and {right}")

    missing_audio = [row["audio_path"] for rows in splits.values() for row in rows if not Path(row["audio_path"]).is_file()]
    if missing_audio:
        errors.append(f"{len(missing_audio)} audio files are missing")

    for source in TARGETS:
        if source in UNRELIABLE_SPEAKER_SOURCES:
            warnings.append(f"{source}: speaker IDs are not reliable enough for leakage enforcement")
            continue
        speakers = {
            split: {
                row["speaker_id"]
                for row in rows
                if row["source_metadata"] == source and row["speaker_id"]
            }
            for split, rows in splits.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = speakers[left] & speakers[right]
            if overlap:
                errors.append(f"{source}: {len(overlap)} speakers overlap between {left} and {right}")

    train_seconds = sum(float(row["duration"]) for row in splits["train"])
    train_hours = train_seconds / 3600.0
    if abs(train_hours - expected_hours) > tolerance_hours:
        errors.append(
            f"training duration {train_hours:.4f}h is outside {expected_hours:.2f}h +/- {tolerance_hours:.2f}h"
        )

    source_seconds = Counter()
    for row in splits["train"]:
        source_seconds[row["source_metadata"]] += float(row["duration"])
    composition = {}
    for source, target in TARGETS.items():
        actual = source_seconds[source] / train_seconds if train_seconds else 0.0
        composition[source] = {"actual": actual, "target": target}
        if abs(actual - target) > 0.015:
            errors.append(f"{source}: actual share {actual:.2%} differs from target {target:.2%} by >1.5pp")

    result = {
        "subset": str(path),
        "status": "ok" if not errors else "failed",
        "train_hours": train_hours,
        "samples": {name: len(rows) for name, rows in splits.items()},
        "composition": composition,
        "errors": errors,
        "warnings": warnings,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LR-search subset manifests.")
    parser.add_argument("--root", type=Path, default=Path("data/lr_search"))
    parser.add_argument("--json-report", type=Path, default=Path("reports/lr_search/subset_validation.json"))
    args = parser.parse_args()

    results = [
        validate_subset(args.root / "coarse_10h", 10.0, 0.02),
        validate_subset(args.root / "main_30h", 30.0, 0.02),
    ]
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if any(item["status"] != "ok" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
