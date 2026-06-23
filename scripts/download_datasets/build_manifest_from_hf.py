#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf

from text_normalization import normalize_uzbek_text


def duration_seconds(path: str) -> float:
    info = sf.info(path)
    return float(info.frames / info.samplerate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical manifest from a saved HF dataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--tier", required=True, choices=["gold", "silver", "bronze"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import load_from_disk

    ds = load_from_disk(str(Path(args.dataset_dir).expanduser()))
    trust = {"gold": 4.0, "silver": 1.5, "bronze": 1.0}[args.tier]
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["audio_path", "text", "duration", "speaker_id", "split", "source_metadata", "dataset_id", "tier", "trust_weight"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, split_ds in ds.items():
            for idx, row in enumerate(split_ds):
                audio = row.get("audio") or {}
                audio_path = audio.get("path") or row.get("audio_path") or row.get("path") or ""
                text = normalize_uzbek_text(row.get("sentence") or row.get("text") or row.get("transcription") or "")
                dur = row.get("duration") or (duration_seconds(audio_path) if audio_path else "")
                writer.writerow(
                    {
                        "audio_path": audio_path,
                        "text": text,
                        "duration": dur,
                        "speaker_id": row.get("client_id") or row.get("speaker_id") or "",
                        "split": split,
                        "source_metadata": json.dumps({"row": idx}, ensure_ascii=False),
                        "dataset_id": args.dataset_id,
                        "tier": args.tier,
                        "trust_weight": trust,
                    }
                )
    print(f"WROTE: {out}")


if __name__ == "__main__":
    main()

