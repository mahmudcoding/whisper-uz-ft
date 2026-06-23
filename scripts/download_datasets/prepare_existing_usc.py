#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from text_normalization import normalize_uzbek_text


def prepare(input_csv: Path, output_csv: Path, split: str | None = None) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as f_in, output_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = [
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
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow(
                {
                    "audio_path": row.get("audio_path", ""),
                    "text": normalize_uzbek_text(row.get("text", "")),
                    "duration": row.get("duration", ""),
                    "speaker_id": row.get("speaker_id", ""),
                    "split": split or row.get("split", ""),
                    "source_metadata": row.get("source_metadata", ""),
                    "dataset_id": "usc",
                    "tier": "gold",
                    "trust_weight": 4.0,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag existing USC manifests with canonical dataset metadata.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    prepare(Path(args.input_csv), Path(args.output_csv), args.split)
    print(f"WROTE: {args.output_csv}")


if __name__ == "__main__":
    main()

