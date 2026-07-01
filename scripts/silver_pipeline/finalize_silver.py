#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
GOLD_FIELDS = [
    "audio_path", "transcript", "dataset_name", "duration_sec",
    "speaker_id", "split", "quality_score",
]
TRAIN_FIELDS = [
    "audio_path", "text", "duration", "speaker_id", "split",
    "source_metadata", "dataset_id", "tier", "trust_weight",
    "quality_score", "sampling_weight",
]


def write_header_only(path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def rejection_counts(paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (
                    row.get("teacher_rejection_reasons")
                    or row.get("rejection_reasons")
                    or row.get("post_gold_rejection_reason")
                    or "unspecified"
                )
                for reason in filter(None, value.split("|")):
                    counts[reason] += 1
    return counts


def quality_band(score: float) -> str:
    return "95-100" if score >= 95 else "90-94" if score >= 90 else "85-89" if score >= 85 else "80-84"


def collect_preserved_silver_rows(master_dir: Path) -> list[dict[str, str]]:
    """Preserve manually governed Silver rows, currently FeruzaSpeech.

    FeruzaSpeech was intentionally moved from Gold to train-only Silver. The
    large-source finalizer rewrites silver_master, so it must carry those rows
    forward instead of dropping them when Kotib-scored rows are finalized.
    """
    detailed_path = master_dir / "silver_manifest_detailed.csv"
    if not detailed_path.exists():
        return []
    preserved: list[dict[str, str]] = []
    with detailed_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row.get("dataset_id") or row.get("dataset_name")
            if dataset == "feruzaspeech":
                preserved.append(row)
    return preserved


def live_gold_indexes(path: Path) -> dict[str, set]:
    indexes: dict[str, set] = {
        "audio": set(),
        "head_duration": set(),
        "text_duration": set(),
        "locked_text": set(),
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("quality_decision") == "reject":
                continue
            if row.get("audio_sha1"):
                indexes["audio"].add(row["audio_sha1"])
            if row.get("audio_head_sha1"):
                indexes["head_duration"].add(
                    (
                        row["audio_head_sha1"],
                        row.get("duration_bucket_100ms", ""),
                    )
                )
            if row.get("transcript_sha1"):
                indexes["text_duration"].add(
                    (
                        row["transcript_sha1"],
                        row.get("duration_bucket_100ms", ""),
                    )
                )
                if row.get("split") in {"val", "validation", "test"}:
                    indexes["locked_text"].add(row["transcript_sha1"])
    return indexes


def gold_overlap_reason(row: dict[str, str], indexes: dict[str, set]) -> str:
    audio_hash = row.get("audio_sha1", "")
    head_key = (
        row.get("audio_head_sha1", ""),
        row.get("duration_bucket_100ms", ""),
    )
    text_key = (
        row.get("transcript_sha1", ""),
        row.get("duration_bucket_100ms", ""),
    )
    if audio_hash and audio_hash in indexes["audio"]:
        return "post_teacher_exact_audio_overlap_gold"
    if head_key[0] and head_key in indexes["head_duration"]:
        return "post_teacher_near_audio_overlap_gold"
    if text_key[0] and text_key in indexes["text_duration"]:
        return "post_teacher_text_duration_overlap_gold"
    if row.get("transcript_sha1") in indexes["locked_text"]:
        return "post_teacher_transcript_overlap_locked_eval"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final train-only SILVER corpus and Gold+Silver curriculum manifests.")
    parser.add_argument("--config", default=str(ROOT / "configs/silver_datasets.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    work_dir = Path(cfg["work_dir"]).expanduser()
    master_dir = Path(cfg["master_dir"]).expanduser()
    report_dir = Path(cfg["report_dir"]).expanduser()
    dedup_dir = Path(cfg["dedup_report_dir"]).expanduser()
    gold_dir = Path(cfg["gold_master_dir"]).expanduser()
    combined_dir = ROOT / "data/gold_silver_training"
    for path in (master_dir, report_dir, dedup_dir, combined_dir):
        path.mkdir(parents=True, exist_ok=True)

    scored_path = work_dir / "silver_teacher_scored.csv"
    preserved_rows = collect_preserved_silver_rows(master_dir)
    gold_indexes = live_gold_indexes(ROOT / "data/gold_work/gold_quality.csv")
    detailed_path = master_dir / "silver_manifest_detailed.csv"
    train_path = master_dir / "train.csv"
    post_gold_rejected_path = report_dir / "post_teacher_gold_overlap_rejected.csv"
    dataset_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "hours": 0.0, "speakers": set(), "quality_scores": []}
    )
    quality_bands: Counter[str] = Counter()
    kept_rows = 0
    kept_hours = 0.0
    with scored_path.open("r", encoding="utf-8", newline="") as input_handle, detailed_path.open(
        "w", encoding="utf-8", newline=""
    ) as detailed_handle, train_path.open("w", encoding="utf-8", newline="") as train_handle, post_gold_rejected_path.open(
        "w", encoding="utf-8", newline=""
    ) as post_gold_rejected_handle:
        reader = csv.DictReader(input_handle)
        detailed_fields = list(
            dict.fromkeys(
                list(reader.fieldnames or [])
                + [field for row in preserved_rows for field in row.keys()]
            )
        )
        detailed_writer = csv.DictWriter(detailed_handle, fieldnames=detailed_fields)
        train_writer = csv.DictWriter(train_handle, fieldnames=GOLD_FIELDS)
        post_gold_fields = detailed_fields + ["post_gold_rejection_reason"]
        post_gold_writer = csv.DictWriter(
            post_gold_rejected_handle, fieldnames=post_gold_fields
        )
        detailed_writer.writeheader()
        train_writer.writeheader()
        post_gold_writer.writeheader()
        for row in preserved_rows:
            detailed_writer.writerow({field: row.get(field, "") for field in detailed_fields})
            score = float(row.get("teacher_quality_score") or row.get("quality_score") or 0.0)
            duration = float(row.get("duration") or row.get("duration_sec") or 0.0)
            dataset_id = row.get("dataset_id") or row.get("dataset_name") or "unknown"
            train_writer.writerow(
                {
                    "audio_path": row["audio_path"],
                    "transcript": row.get("normalized_text") or row.get("transcript") or row.get("text") or "",
                    "dataset_name": dataset_id,
                    "duration_sec": duration,
                    "speaker_id": row.get("speaker_id", ""),
                    "split": "train",
                    "quality_score": score,
                }
            )
            stats = dataset_stats[dataset_id]
            stats["rows"] += 1
            stats["hours"] += duration / 3600
            if row.get("speaker_id"):
                stats["speakers"].add(row["speaker_id"])
            stats["quality_scores"].append(score)
            quality_bands[quality_band(score)] += 1
            kept_rows += 1
            kept_hours += duration / 3600
        for row in reader:
            if row.get("teacher_decision") != "keep":
                continue
            overlap_reason = gold_overlap_reason(row, gold_indexes)
            if overlap_reason:
                post_gold_writer.writerow(
                    {**row, "post_gold_rejection_reason": overlap_reason}
                )
                continue
            detailed_writer.writerow(row)
            score = float(row["teacher_quality_score"])
            duration = float(row["duration"])
            dataset_id = row["dataset_id"]
            train_writer.writerow(
                {
                    "audio_path": row["audio_path"],
                    "transcript": row["normalized_text"],
                    "dataset_name": dataset_id,
                    "duration_sec": duration,
                    "speaker_id": row.get("speaker_id", ""),
                    "split": "train",
                    "quality_score": score,
                }
            )
            stats = dataset_stats[dataset_id]
            stats["rows"] += 1
            stats["hours"] += duration / 3600
            if row.get("speaker_id"):
                stats["speakers"].add(row["speaker_id"])
            stats["quality_scores"].append(score)
            quality_bands[quality_band(score)] += 1
            kept_rows += 1
            kept_hours += duration / 3600
    write_header_only(master_dir / "val.csv", GOLD_FIELDS)
    write_header_only(master_dir / "test.csv", GOLD_FIELDS)

    with (combined_dir / "train.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TRAIN_FIELDS)
        writer.writeheader()
        with (gold_dir / "train.csv").open("r", encoding="utf-8", newline="") as gold_handle:
            for row in csv.DictReader(gold_handle):
                quality = float(row["quality_score"])
                writer.writerow(
                    {
                        "audio_path": row["audio_path"],
                        "text": row["transcript"],
                        "duration": row["duration_sec"],
                        "speaker_id": row["speaker_id"],
                        "split": "train",
                        "source_metadata": "{}",
                        "dataset_id": row["dataset_name"],
                        "tier": "gold",
                        "trust_weight": 4.0,
                        "quality_score": quality,
                        "sampling_weight": 4.0 * min(1.25, quality / 100),
                    }
                )
        with detailed_path.open("r", encoding="utf-8", newline="") as silver_handle:
            for row in csv.DictReader(silver_handle):
                quality = float(row.get("teacher_quality_score") or row.get("quality_score") or 0.0)
                writer.writerow(
                    {
                        "audio_path": row["audio_path"],
                        "text": row.get("normalized_text") or row.get("transcript") or row.get("text") or "",
                        "duration": row.get("duration") or row.get("duration_sec") or "",
                        "speaker_id": row.get("speaker_id", ""),
                        "split": "train",
                        "source_metadata": row.get("source_metadata", "{}"),
                        "dataset_id": row.get("dataset_id") or row.get("dataset_name") or "unknown",
                        "tier": "silver",
                        "trust_weight": 1.5,
                        "quality_score": quality,
                        "sampling_weight": 1.5 * min(1.25, quality / 100),
                    }
                )
    for split in ("val", "test"):
        with (gold_dir / f"{split}.csv").open("r", encoding="utf-8", newline="") as source, (
            combined_dir / f"{split}.csv"
        ).open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=TRAIN_FIELDS)
            writer.writeheader()
            for row in csv.DictReader(source):
                quality = float(row["quality_score"])
                writer.writerow(
                    {
                        "audio_path": row["audio_path"],
                        "text": row["transcript"],
                        "duration": row["duration_sec"],
                        "speaker_id": row["speaker_id"],
                        "split": split,
                        "source_metadata": "{}",
                        "dataset_id": row["dataset_name"],
                        "tier": "gold",
                        "trust_weight": 4.0,
                        "quality_score": quality,
                        "sampling_weight": 1.0,
                    }
                )

    serialized_stats = {}
    for name, values in dataset_stats.items():
        scores = values.pop("quality_scores")
        speakers = values.pop("speakers")
        serialized_stats[name] = {
            **values,
            "speakers": len(speakers),
            "mean_quality_score": sum(scores) / len(scores) if scores else 0.0,
        }
    reasons = rejection_counts(
        [
            report_dir / "prefilter_rejected.csv",
            report_dir / "teacher_rejected.csv",
            post_gold_rejected_path,
        ]
    )
    summary = {
        "final_unique_rows": kept_rows,
        "final_unique_hours": kept_hours,
        "final_by_dataset": serialized_stats,
        "quality_bands": dict(quality_bands),
        "rejection_reasons": dict(reasons.most_common()),
        "evaluation_policy": "SILVER is train-only; Gold validation and test remain unchanged.",
        "paths": {
            "silver_train": str(train_path),
            "silver_detailed": str(detailed_path),
            "gold_silver_training": str(combined_dir),
        },
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dedup_reasons = {key: value for key, value in reasons.items() if "duplicate" in key or "overlap" in key}
    (dedup_dir / "summary.json").write_text(json.dumps(dedup_reasons, indent=2), encoding="utf-8")

    lines = [
        "# SILVER Corpus Report", "",
        "## Final Corpus", "",
        f"- Training rows: **{kept_rows:,}**",
        f"- Usable hours: **{kept_hours:.2f}**",
        "- Evaluation data: Gold validation/test only; no SILVER rows enter model selection or final evaluation.",
        "", "## Per Dataset", "",
        "| Dataset | Rows | Hours | Known speakers | Mean quality |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in serialized_stats.items():
        lines.append(
            f"| {name} | {values['rows']:,} | {values['hours']:.2f} | "
            f"{values['speakers']:,} | {values['mean_quality_score']:.2f} |"
        )
    lines.extend(["", "## Rejection Reasons", ""])
    for reason, count in reasons.most_common():
        lines.append(f"- `{reason}`: {count:,}")
    lines.extend(
        [
            "", "## Training Integration", "",
            f"- SILVER governance manifest: `{train_path}`",
            f"- Detailed quality manifest: `{detailed_path}`",
            f"- Gold+Silver curriculum manifests: `{combined_dir}`",
            "- Initial sampling weights: Gold 4.0, SILVER 1.5, quality-scaled.",
        ]
    )
    (report_dir / "SILVER_CORPUS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
