from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def find_overlap(input_csv: Path, output_csv: Path) -> None:
    rows: list[dict] = []
    by_audio: dict[str, list[int]] = defaultdict(list)
    by_head: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_text_duration: dict[tuple[str, str], list[int]] = defaultdict(list)

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for idx, row in enumerate(reader):
            rows.append(row)
            if row.get("audio_sha1"):
                by_audio[row["audio_sha1"]].append(idx)
            if row.get("audio_head_sha1"):
                by_head[(row.get("audio_head_sha1", ""), row.get("duration_bucket_100ms", ""))].append(idx)
            if row.get("transcript_sha1"):
                by_text_duration[(row.get("transcript_sha1", ""), row.get("duration_bucket_100ms", ""))].append(idx)

    fields = list(dict.fromkeys(fields + ["duplicate_audio", "near_duplicate_audio", "text_duration_overlap", "overlap_group"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            exact = by_audio.get(row.get("audio_sha1", ""), [])
            near = by_head.get((row.get("audio_head_sha1", ""), row.get("duration_bucket_100ms", "")), [])
            text_dur = by_text_duration.get((row.get("transcript_sha1", ""), row.get("duration_bucket_100ms", "")), [])
            group = exact or near or text_dur
            row["duplicate_audio"] = len(exact) > 1
            row["near_duplicate_audio"] = len(near) > 1
            row["text_duration_overlap"] = len(text_dur) > 1
            row["overlap_group"] = "|".join(str(i) for i in group[:20]) if len(group) > 1 else ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    find_overlap(Path(args.input_csv), Path(args.output_csv))


if __name__ == "__main__":
    main()

