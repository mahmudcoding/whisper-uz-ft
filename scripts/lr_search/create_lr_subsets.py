#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SOURCE_WEIGHTS = {
    "usc": 0.50,
    "common_voice_uz": 0.40,
    "fleurs_uz": 0.10,
}
OUTPUT_COLUMNS = [
    "audio_path",
    "text",
    "duration",
    "speaker_id",
    "split",
    "source_metadata",
]


def stable_key(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def read_gold(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"audio_path", "transcript", "dataset_name", "duration_sec", "speaker_id", "split"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for row in rows:
        row["duration_value"] = float(row["duration_sec"])
        row["text_length"] = len(row["transcript"].split())
    return rows


def quantile_boundaries(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, math.floor(len(ordered) * i / bins))] for i in range(1, bins)]


def bin_index(value: float, boundaries: list[float]) -> int:
    return sum(value > boundary for boundary in boundaries)


def stratified_duration_sample(
    rows: list[dict[str, str]],
    target_seconds: float,
    seed: int,
    source: str,
) -> list[dict[str, str]]:
    if not rows:
        raise ValueError(f"No rows available for source {source}")
    duration_bounds = quantile_boundaries([float(r["duration_value"]) for r in rows], 5)
    text_bounds = quantile_boundaries([float(r["text_length"]) for r in rows], 4)
    strata: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            bin_index(float(row["duration_value"]), duration_bounds),
            bin_index(float(row["text_length"]), text_bounds),
        )
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(key=lambda r: stable_key(seed, source, key, r["audio_path"]))

    total_seconds = sum(float(r["duration_value"]) for r in rows)
    selected: list[dict[str, str]] = []
    selected_paths: set[str] = set()
    for key in sorted(strata):
        stratum = strata[key]
        stratum_seconds = sum(float(r["duration_value"]) for r in stratum)
        stratum_target = target_seconds * stratum_seconds / total_seconds
        running = 0.0
        stratum_selected: list[dict[str, str]] = []
        for row in stratum:
            if running >= stratum_target:
                break
            stratum_selected.append(row)
            running += float(row["duration_value"])
        if stratum_selected:
            last = stratum_selected[-1]
            without_last = running - float(last["duration_value"])
            if abs(without_last - stratum_target) < abs(running - stratum_target):
                stratum_selected.pop()
        selected.extend(stratum_selected)
        selected_paths.update(row["audio_path"] for row in stratum_selected)

    current = sum(float(r["duration_value"]) for r in selected)
    remaining = sorted(
        (r for r in rows if r["audio_path"] not in selected_paths),
        key=lambda r: stable_key(seed, source, "fill", r["audio_path"]),
    )
    for row in remaining:
        if current >= target_seconds:
            break
        selected.append(row)
        selected_paths.add(row["audio_path"])
        current += float(row["duration_value"])

    # Remove an overshooting final row only when doing so gets closer to target.
    if selected:
        last = selected[-1]
        without_last = current - float(last["duration_value"])
        if abs(without_last - target_seconds) < abs(current - target_seconds):
            selected.pop()
    return sorted(selected, key=lambda r: (r["dataset_name"], r["audio_path"]))


def sample_split(
    rows: list[dict[str, str]],
    target_hours: float,
    seed: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for source, proportion in SOURCE_WEIGHTS.items():
        source_rows = [row for row in rows if row["dataset_name"] == source]
        selected.extend(
            stratified_duration_sample(
                source_rows,
                target_seconds=target_hours * 3600.0 * proportion,
                seed=seed,
                source=source,
            )
        )
    return selected


def to_training_row(row: dict[str, str], split: str) -> dict[str, object]:
    return {
        "audio_path": row["audio_path"],
        "text": row["transcript"],
        "duration": float(row["duration_value"]),
        "speaker_id": row["speaker_id"],
        "split": split,
        "source_metadata": row["dataset_name"],
    }


def write_manifest(path: Path, rows: Iterable[dict[str, str]], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_training_row(row, split))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    durations = [float(row["duration_value"]) for row in rows]
    source_seconds = Counter()
    source_samples = Counter()
    for row in rows:
        source_seconds[row["dataset_name"]] += float(row["duration_value"])
        source_samples[row["dataset_name"]] += 1
    return {
        "samples": len(rows),
        "hours": sum(durations) / 3600.0,
        "speakers": len({row["speaker_id"] for row in rows if row["speaker_id"]}),
        "average_duration_sec": sum(durations) / len(durations),
        "min_duration_sec": min(durations),
        "max_duration_sec": max(durations),
        "source_hours": {key: source_seconds[key] / 3600.0 for key in SOURCE_WEIGHTS},
        "source_samples": dict(source_samples),
    }


def write_report(path: Path, name: str, seed: int, split_rows: dict[str, list[dict[str, str]]]) -> None:
    train_summary = summarize(split_rows["train"])
    lines = [
        f"# {name} LR Search Subset",
        "",
        f"- Deterministic seed: `{seed}`",
        f"- Training hours: `{train_summary['hours']:.4f}`",
        f"- Training samples: `{train_summary['samples']}`",
        f"- Training speakers: `{train_summary['speakers']}`",
        f"- Average duration: `{train_summary['average_duration_sec']:.3f}s`",
        f"- Duration range: `{train_summary['min_duration_sec']:.3f}s` to `{train_summary['max_duration_sec']:.3f}s`",
        "",
        "## Training Composition",
        "",
        "| Dataset | Hours | Share | Samples | Target share |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, target in SOURCE_WEIGHTS.items():
        hours = train_summary["source_hours"][source]
        share = hours / train_summary["hours"] if train_summary["hours"] else 0.0
        lines.append(
            f"| {source} | {hours:.4f} | {share:.2%} | "
            f"{train_summary['source_samples'].get(source, 0)} | {target:.0%} |"
        )
    lines.extend(["", "## Evaluation Manifests", ""])
    for split in ("val", "test"):
        summary = summarize(split_rows[split])
        lines.append(
            f"- `{split}.csv`: {summary['hours']:.4f}h, {summary['samples']} samples, "
            f"{summary['speakers']} recorded speaker IDs"
        )
    lines.extend(
        [
            "",
            "## Sampling Method",
            "",
            "Rows are selected only from the matching Gold split. Within each source, sampling is",
            "stratified jointly by duration quintile and transcript-word-count quartile. Stable",
            "SHA-256 ordering makes selection reproducible and independent of input row order.",
            "The validator enforces path separation and reliable speaker separation across splits.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def create_subset(
    gold_dir: Path,
    output_dir: Path,
    report_path: Path,
    train_hours: float,
    eval_hours: float,
    seed: int,
    name: str,
) -> None:
    gold = {
        split: read_gold(gold_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    selected = {
        "train": sample_split(gold["train"], train_hours, seed),
        "val": sample_split(gold["val"], eval_hours, seed + 1),
        "test": sample_split(gold["test"], eval_hours, seed + 2),
    }
    for split, rows in selected.items():
        write_manifest(output_dir / f"{split}.csv", rows, split)
    write_report(report_path, name, seed, selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Gold-corpus LR-search subsets.")
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold_master"))
    parser.add_argument("--output-root", type=Path, default=Path("data/lr_search"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/lr_search"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--eval-hours", type=float, default=1.0)
    args = parser.parse_args()

    jobs = [
        ("coarse_10h", 10.0, "subset_10h_report.md"),
        ("main_30h", 30.0, "subset_30h_report.md"),
    ]
    for index, (name, hours, report_name) in enumerate(jobs):
        create_subset(
            args.gold_dir,
            args.output_root / name,
            args.report_root / report_name,
            train_hours=hours,
            eval_hours=args.eval_hours,
            seed=args.seed + index * 100,
            name=name,
        )
        print(f"Created {name} at {args.output_root / name}")


if __name__ == "__main__":
    main()
