#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare externally collected YouTube Uzbek speech manifests.")
    parser.add_argument("--input-manifest", required=True, help="CSV/TSV with at least audio_path,text,duration.")
    parser.add_argument("--dataset-id", required=True, choices=["it_youtube_uz", "news_youtube_uz", "podcasts_tashkent"])
    parser.add_argument("--tier", default="silver", choices=["gold", "silver", "bronze"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input_manifest).expanduser()
    dst = Path(args.output).expanduser()
    delimiter = "\t" if src.suffix.lower() == ".tsv" else ","
    trust = {"gold": 4.0, "silver": 1.5, "bronze": 1.0}[args.tier]
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8", newline="") as f_in, dst.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in, delimiter=delimiter)
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
        for idx, row in enumerate(reader):
            writer.writerow(
                {
                    "audio_path": row.get("audio_path") or row.get("path") or row.get("audio") or "",
                    "text": row.get("text") or row.get("sentence") or row.get("transcript") or "",
                    "duration": row.get("duration") or "",
                    "speaker_id": row.get("speaker_id") or row.get("speaker") or "",
                    "split": row.get("split") or "train",
                    "source_metadata": row.get("source_metadata") or f"row={idx}",
                    "dataset_id": args.dataset_id,
                    "tier": args.tier,
                    "trust_weight": trust,
                }
            )
    print(f"WROTE: {dst}")


if __name__ == "__main__":
    main()

