from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

from text_normalization import normalize_uzbek_text


def text_hash(text: str) -> str:
    norm = normalize_uzbek_text(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def flag_transcript_duplicates(input_csv: Path, output_csv: Path, near_threshold: float = 96.0) -> None:
    rows: list[dict] = []
    groups: dict[str, list[int]] = defaultdict(list)
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for idx, row in enumerate(reader):
            norm = normalize_uzbek_text(row.get("text", ""))
            row["normalized_text"] = norm
            row["transcript_sha1"] = hashlib.sha1(norm.encode("utf-8")).hexdigest()
            groups[row["transcript_sha1"]].append(idx)
            rows.append(row)

    near_flags = set()
    if fuzz is not None:
        by_prefix: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            norm = row["normalized_text"]
            by_prefix[norm[:12]].append(idx)
        for idxs in by_prefix.values():
            for i, left_idx in enumerate(idxs):
                for right_idx in idxs[i + 1 :]:
                    if fuzz.ratio(rows[left_idx]["normalized_text"], rows[right_idx]["normalized_text"]) >= near_threshold:
                        near_flags.add(left_idx)
                        near_flags.add(right_idx)

    fields = list(dict.fromkeys(fields + ["normalized_text", "transcript_sha1", "duplicate_transcript", "near_duplicate_transcript"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            row["duplicate_transcript"] = len(groups[row["transcript_sha1"]]) > 1
            row["near_duplicate_transcript"] = idx in near_flags
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--near-threshold", type=float, default=96.0)
    args = parser.parse_args()
    flag_transcript_duplicates(Path(args.input_csv), Path(args.output_csv), args.near_threshold)


if __name__ == "__main__":
    main()

