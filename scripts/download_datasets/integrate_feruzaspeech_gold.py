#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from data_quality.scoring import score_manifest
from dedup.audio_hash import add_audio_hashes
from dedup.transcript_dedup import flag_transcript_duplicates


ROOT = Path(__file__).resolve().parents[2]
MASTER_FIELDS = [
    "audio_path",
    "transcript",
    "dataset_name",
    "duration_sec",
    "speaker_id",
    "split",
    "quality_score",
]
TRAINING_FIELDS = [
    "audio_path",
    "text",
    "duration",
    "speaker_id",
    "split",
    "source_metadata",
]
UNRELIABLE_SPEAKER_DATASETS = {"fleurs_uz"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def prepare_feruza_quality(feruza_dir: Path, work_dir: Path) -> list[dict[str, str]]:
    canonical_rows: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        canonical_rows.extend(
            read_rows(feruza_dir / "manifests" / f"{split}_canonical.csv")
        )
    canonical = work_dir / "feruza_canonical.csv"
    text_dedup = work_dir / "feruza_text_dedup.csv"
    hashes = work_dir / "feruza_hashes.csv"
    overlap = work_dir / "feruza_overlap.csv"
    quality = work_dir / "feruza_quality.csv"
    write_rows(canonical, canonical_rows)
    flag_transcript_duplicates(canonical, text_dedup)
    add_audio_hashes(text_dedup, hashes, sample_seconds=10.0)

    # Quality scoring expects overlap fields, but the combined-corpus flags are
    # assigned below after existing Gold and Feruza rows are merged.
    hash_rows = read_rows(hashes)
    for row in hash_rows:
        row.update(
            {
                "duplicate_audio": "False",
                "near_duplicate_audio": "False",
                "text_duration_overlap": "False",
                "overlap_group": "",
            }
        )
    write_rows(overlap, hash_rows)
    score_manifest(overlap, quality)
    return read_rows(quality)


def existing_master_details(
    gold_master: Path, existing_quality: Path
) -> list[dict[str, str]]:
    details = {row["audio_path"]: row for row in read_rows(existing_quality)}
    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for split_file, governed_split in (
        ("train.csv", "train"),
        ("val.csv", "val"),
        ("test.csv", "test"),
    ):
        for master_row in read_rows(gold_master / split_file):
            if master_row.get("dataset_name") == "feruzaspeech":
                continue
            row = details.get(master_row["audio_path"])
            if row is None:
                missing.append(master_row["audio_path"])
            else:
                selected_row = dict(row)
                selected_row["split"] = governed_split
                selected.append(selected_row)
    if missing:
        raise RuntimeError(
            f"{len(missing)} existing Gold rows lack detailed quality records; "
            f"first={missing[0]}"
        )
    return selected


def assign_combined_overlap(rows: list[dict[str, str]]) -> None:
    by_audio: dict[str, list[int]] = defaultdict(list)
    by_head: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_text_duration: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_text: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("audio_sha1"):
            by_audio[row["audio_sha1"]].append(index)
        if row.get("audio_head_sha1"):
            by_head[
                (row["audio_head_sha1"], row.get("duration_bucket_100ms", ""))
            ].append(index)
        if row.get("transcript_sha1"):
            by_text[row["transcript_sha1"]].append(index)
            by_text_duration[
                (row["transcript_sha1"], row.get("duration_bucket_100ms", ""))
            ].append(index)

    for index, row in enumerate(rows):
        exact = by_audio.get(row.get("audio_sha1", ""), [])
        near = by_head.get(
            (row.get("audio_head_sha1", ""), row.get("duration_bucket_100ms", "")),
            [],
        )
        text_duration = by_text_duration.get(
            (row.get("transcript_sha1", ""), row.get("duration_bucket_100ms", "")),
            [],
        )
        transcript = by_text.get(row.get("transcript_sha1", ""), [])
        row["duplicate_audio"] = str(len(exact) > 1)
        row["near_duplicate_audio"] = str(len(near) > 1)
        row["text_duration_overlap"] = str(len(text_duration) > 1)
        row["duplicate_transcript"] = str(len(transcript) > 1)
        group = sorted(set(exact + near + text_duration))
        row["overlap_group"] = "|".join(map(str, group[:20])) if len(group) > 1 else ""


def select_final(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    eligible = [
        index
        for index, row in enumerate(rows)
        if row.get("quality_decision") != "reject"
    ]
    parent = {index: index for index in eligible}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    exact_groups: dict[str, list[int]] = defaultdict(list)
    near_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in eligible:
        row = rows[index]
        if row.get("audio_sha1"):
            exact_groups[row["audio_sha1"]].append(index)
        if row.get("audio_head_sha1"):
            near_groups[
                (row["audio_head_sha1"], row.get("duration_bucket_100ms", ""))
            ].append(index)
    for group in list(exact_groups.values()) + list(near_groups.values()):
        for index in group[1:]:
            union(group[0], index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        components[find(index)].append(index)

    split_priority = {"test": 3, "val": 2, "validation": 2, "train": 1}
    keep_indexes: set[int] = set()
    removals: list[dict] = []
    for component in components.values():
        if len(component) == 1:
            keep_indexes.add(component[0])
            continue

        # Existing Gold is already validated and its locked split identity has
        # priority over newly added Feruza rows. Within Feruza, protect test,
        # then validation, then train.
        winner = max(
            component,
            key=lambda index: (
                rows[index].get("dataset_id") != "feruzaspeech",
                split_priority.get(rows[index].get("split", ""), 0),
                float(rows[index].get("quality_score") or 0),
                -index,
            ),
        )
        keep_indexes.add(winner)
        for index in component:
            if index == winner:
                continue
            removals.append(
                {
                    "audio_path": rows[index]["audio_path"],
                    "dataset_id": rows[index]["dataset_id"],
                    "split": rows[index]["split"],
                    "duration": rows[index]["duration"],
                    "quality_score": rows[index].get("quality_score", ""),
                    "kept_audio_path": rows[winner]["audio_path"],
                    "kept_dataset_id": rows[winner]["dataset_id"],
                    "audio_sha1": rows[index].get("audio_sha1", ""),
                    "audio_head_sha1": rows[index].get("audio_head_sha1", ""),
                    "reason": "exact_or_near_audio_duplicate",
                    "text": rows[index]["text"],
                }
            )
    return [rows[index] for index in sorted(keep_indexes)], removals


def master_row(row: dict[str, str]) -> dict:
    return {
        "audio_path": row["audio_path"],
        "transcript": row["normalized_text"] or row["text"],
        "dataset_name": row["dataset_id"],
        "duration_sec": row["duration"],
        "speaker_id": row.get("speaker_id", ""),
        "split": row["split"],
        "quality_score": row.get("quality_score", "100"),
    }


def training_row(row: dict[str, str]) -> dict:
    metadata = {
        "dataset_id": row["dataset_id"],
        "tier": row.get("tier", "gold"),
        "quality_score": float(row.get("quality_score") or 100.0),
    }
    return {
        "audio_path": row["audio_path"],
        "text": row["normalized_text"] or row["text"],
        "duration": row["duration"],
        "speaker_id": row.get("speaker_id", ""),
        "split": row["split"],
        "source_metadata": json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ),
    }


def stats_by_dataset(rows: list[dict]) -> dict:
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "rows": 0,
            "hours": 0.0,
            "speakers": set(),
            "quality_classes": Counter(),
        }
    )
    for row in rows:
        item = stats[row["dataset_id"]]
        item["rows"] += 1
        item["hours"] += float(row["duration"]) / 3600.0
        if row.get("speaker_id"):
            item["speakers"].add(row["speaker_id"])
        item["quality_classes"][row.get("quality_class", "unknown")] += 1
    return {
        name: {
            "rows": values["rows"],
            "hours": values["hours"],
            "speakers": len(values["speakers"]),
            "quality_classes": dict(values["quality_classes"]),
        }
        for name, values in stats.items()
    }


def validate_final(rows: list[dict]) -> dict:
    path_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    split_summary: dict[str, dict] = defaultdict(
        lambda: {"rows": 0, "hours": 0.0, "datasets": Counter()}
    )
    for row in rows:
        split = row["split"]
        path_splits[row["audio_path"]].add(split)
        if row.get("audio_sha1"):
            hash_splits[row["audio_sha1"]].add(split)
        if (
            row.get("speaker_id")
            and row["dataset_id"] not in UNRELIABLE_SPEAKER_DATASETS
        ):
            speaker_splits[f"{row['dataset_id']}:{row['speaker_id']}"].add(split)
        if not Path(row["audio_path"]).is_file():
            missing.append(row["audio_path"])
        item = split_summary[split]
        item["rows"] += 1
        item["hours"] += float(row["duration"]) / 3600.0
        item["datasets"][row["dataset_id"]] += 1
    return {
        "total_rows": len(rows),
        "total_hours": sum(float(row["duration"]) for row in rows) / 3600.0,
        "missing_audio_paths": len(missing),
        "path_leakage_across_splits": sum(
            len(splits) > 1 for splits in path_splits.values()
        ),
        "content_hash_leakage_across_splits": sum(
            len(splits) > 1 for splits in hash_splits.values()
        ),
        "known_speaker_leakage_across_splits": sum(
            len(splits) > 1 for splits in speaker_splits.values()
        ),
        "split_summary": {
            split: {
                "rows": values["rows"],
                "hours": values["hours"],
                "datasets": dict(values["datasets"]),
            }
            for split, values in split_summary.items()
        },
        "examples": {"missing_audio_paths": missing[:20]},
    }


def integrate(
    feruza_dir: Path,
    gold_master: Path,
    work_dir: Path,
    quality_report_dir: Path,
    dedup_report_dir: Path,
    training_schema_dir: Path,
) -> dict:
    old_quality = work_dir / "gold_quality.csv"
    existing = existing_master_details(gold_master, old_quality)
    feruza = prepare_feruza_quality(feruza_dir, work_dir)
    combined = existing + feruza
    assign_combined_overlap(combined)

    raw_fields = [
        "audio_path",
        "text",
        "duration",
        "speaker_id",
        "split",
        "source_metadata",
        "dataset_id",
        "tier",
        "trust_weight",
    ]
    write_rows(work_dir / "gold_raw_combined.csv", combined, raw_fields)
    write_rows(work_dir / "gold_text_dedup.csv", combined)
    write_rows(work_dir / "gold_hashes.csv", combined)
    write_rows(work_dir / "gold_overlap.csv", combined)
    write_rows(work_dir / "gold_quality.csv", combined)

    final_rows, duplicate_removals = select_final(combined)
    final_rows.sort(
        key=lambda row: (
            {"train": 0, "val": 1, "test": 2}.get(row["split"], 9),
            row["dataset_id"],
            row["audio_path"],
        )
    )
    suspicious_or_rejected = []
    for row in combined:
        score = float(row.get("quality_score") or 0)
        reasons = row.get("quality_reasons", "")
        if row.get("quality_decision") == "reject" or score < 80 or reasons:
            suspicious_or_rejected.append(
                {
                    "audio_path": row["audio_path"],
                    "dataset_id": row["dataset_id"],
                    "duration": row["duration"],
                    "quality_score": score,
                    "quality_class": row.get("quality_class", ""),
                    "decision": (
                        "reject"
                        if row.get("quality_decision") == "reject"
                        else "suspicious"
                    ),
                    "quality_reasons": reasons,
                    "text": row["text"],
                }
            )
    write_rows(
        quality_report_dir / "rejected_or_suspicious.csv",
        suspicious_or_rejected,
    )
    write_rows(
        dedup_report_dir / "duplicate_removals_recommended.csv",
        duplicate_removals,
    )

    validation = validate_final(final_rows)
    if any(
        validation[key]
        for key in (
            "missing_audio_paths",
            "path_leakage_across_splits",
            "content_hash_leakage_across_splits",
            "known_speaker_leakage_across_splits",
        )
    ):
        raise RuntimeError(f"Final Gold validation failed: {validation}")

    for split, filename in (
        ("train", "train.csv"),
        ("val", "val.csv"),
        ("test", "test.csv"),
    ):
        write_rows(
            gold_master / filename,
            [master_row(row) for row in final_rows if row["split"] == split],
            MASTER_FIELDS,
        )
        write_rows(
            training_schema_dir / filename,
            [training_row(row) for row in final_rows if row["split"] == split],
            TRAINING_FIELDS,
        )

    quality_rejects = sum(
        row.get("quality_decision") == "reject" for row in combined
    )
    summary = {
        "raw_rows": len(combined),
        "raw_hours": sum(float(row["duration"]) for row in combined) / 3600.0,
        "quality_reject_rows": quality_rejects,
        "duplicate_removed_rows": len(duplicate_removals),
        "final_unique_rows": len(final_rows),
        "final_unique_hours": validation["total_hours"],
        "raw_by_dataset": stats_by_dataset(combined),
        "final_by_dataset": stats_by_dataset(final_rows),
        "splits": validation["split_summary"],
        "duplicate_stats": {
            "removed_audio_duplicate_or_near_duplicate_rows": len(
                duplicate_removals
            ),
            "duplicate_audio_rows_flagged": sum(
                bool_value(row.get("duplicate_audio")) for row in combined
            ),
            "near_duplicate_audio_rows_flagged": sum(
                bool_value(row.get("near_duplicate_audio")) for row in combined
            ),
            "duplicate_transcript_rows_flagged": sum(
                bool_value(row.get("duplicate_transcript")) for row in combined
            ),
            "text_duration_overlap_rows_flagged": sum(
                bool_value(row.get("text_duration_overlap")) for row in combined
            ),
        },
        "license_warning": (
            "FeruzaSpeech is governed by gated K2Speech terms for academic "
            "research/internal use; review rights before commercial use."
        ),
    }
    quality_report_dir.mkdir(parents=True, exist_ok=True)
    dedup_report_dir.mkdir(parents=True, exist_ok=True)
    (quality_report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (quality_report_dir / "master_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    (dedup_report_dir / "summary.json").write_text(
        json.dumps(summary["duplicate_stats"], indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Gold Corpus Report",
        "",
        f"- Final rows: {len(final_rows):,}",
        f"- Final hours: {validation['total_hours']:.4f}",
        "- Missing audio: 0",
        "- Cross-split path leakage: 0",
        "- Cross-split audio-hash leakage: 0",
        "- Known speaker leakage: 0",
        "",
        "| Dataset | Rows | Hours | Known speakers |",
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["final_by_dataset"].items():
        report.append(
            f"| {name} | {values['rows']:,} | {values['hours']:.4f} | "
            f"{values['speakers']:,} |"
        )
    report.extend(
        [
            "",
            "FeruzaSpeech uses its official speaker-disjoint train/dev/test split. "
            "Clips over 30 seconds were excluded before integration because aligned "
            "chunk timestamps are unavailable.",
            "",
            "FeruzaSpeech licensing is restrictive: academic research/internal use "
            "only under the gated K2Speech terms. Do not redistribute the source or "
            "prepared audio, and review rights before commercial deployment.",
        ]
    )
    Path(quality_report_dir / "GOLD_CORPUS_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate prepared FeruzaSpeech into the validated Gold corpus."
    )
    parser.add_argument(
        "--feruza-dir",
        type=Path,
        default=Path("/home/mahmud/datasets/feruzaspeech"),
    )
    parser.add_argument("--gold-master", type=Path, default=ROOT / "data/gold_master")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/gold_work")
    parser.add_argument(
        "--quality-report-dir",
        type=Path,
        default=ROOT / "reports/gold_quality_report",
    )
    parser.add_argument(
        "--dedup-report-dir",
        type=Path,
        default=ROOT / "reports/gold_dedup_report",
    )
    parser.add_argument(
        "--training-schema-dir",
        type=Path,
        default=ROOT / "data/gold_master_training_schema",
    )
    args = parser.parse_args()
    summary = integrate(
        args.feruza_dir.expanduser().resolve(),
        args.gold_master.expanduser().resolve(),
        args.work_dir.expanduser().resolve(),
        args.quality_report_dir.expanduser().resolve(),
        args.dedup_report_dir.expanduser().resolve(),
        args.training_schema_dir.expanduser().resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
