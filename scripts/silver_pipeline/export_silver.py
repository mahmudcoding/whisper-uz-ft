#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import yaml
from datasets import Audio, load_dataset

from text_normalization import normalize_uzbek_text


ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
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


def decode_audio(payload: dict[str, Any]) -> tuple[np.ndarray, int, str]:
    raw = payload.get("bytes")
    path = payload.get("path")
    if raw:
        source_name = str(path or "embedded_audio")
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            process = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-f",
                    "f32le",
                    "-acodec",
                    "pcm_f32le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "pipe:1",
                ],
                input=raw,
                capture_output=True,
                check=True,
            )
            audio = np.frombuffer(process.stdout, dtype="<f4")
            sr = 16000
        return np.asarray(audio, dtype=np.float32), int(sr), source_name
    if path:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr), str(path)
    raise ValueError("Audio payload has neither bytes nor path")


def normalize_audio(payload: dict[str, Any]) -> tuple[np.ndarray, str]:
    audio, sr, source_name = decode_audio(payload)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if not audio.size:
        raise ValueError("empty audio")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio, source_name


def parquet_files(raw_dir: Path, split: str) -> list[str]:
    files = sorted(raw_dir.glob(f"data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files under {raw_dir / 'data'}")
    return [str(path) for path in files]


def export_split(name: str, spec: dict[str, Any], output_root: Path, split: str) -> dict[str, Any]:
    dataset_root = output_root / name
    raw_dir = dataset_root / "raw_hf"
    processed_dir = dataset_root / "processed"
    audio_root = processed_dir / "audio" / split
    manifest_dir = processed_dir / "manifests"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{split}_canonical.csv"
    rejected_path = manifest_dir / f"{split}_export_rejected.csv"
    progress_path = manifest_dir / f"{split}_export_progress.json"

    data = load_dataset("parquet", data_files=parquet_files(raw_dir, split), split="train")
    audio_column = spec["audio_column"]
    data = data.cast_column(audio_column, Audio(decode=False))
    text_column = spec["text_column"]
    speaker_column = spec.get("speaker_column") or ""
    completed_source_rows = 0
    if progress_path.exists():
        completed_source_rows = int(
            json.loads(progress_path.read_text(encoding="utf-8")).get("completed_source_rows", 0)
        )
    existing_rows = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = max(0, sum(1 for _ in handle) - 1)
    mode = "a" if completed_source_rows else "w"
    reject_mode = "a" if rejected_path.exists() else "w"
    rows = existing_rows
    hours = 0.0
    rejected = 0
    with manifest_path.open(mode, encoding="utf-8", newline="") as manifest_handle, rejected_path.open(
        reject_mode, encoding="utf-8", newline=""
    ) as reject_handle:
        writer = csv.DictWriter(manifest_handle, fieldnames=FIELDS)
        reject_fields = ["dataset_id", "split", "source_index", "reason"]
        reject_writer = csv.DictWriter(reject_handle, fieldnames=reject_fields)
        if not completed_source_rows:
            writer.writeheader()
        if reject_mode == "w":
            reject_writer.writeheader()
        for idx, row in enumerate(data):
            if idx < completed_source_rows:
                continue
            try:
                text = normalize_uzbek_text(row.get(text_column))
                if not text:
                    raise ValueError("empty normalized transcript")
                audio, source_name = normalize_audio(row[audio_column])
                duration = float(len(audio) / 16000)
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"invalid duration {duration}")
                shard_dir = audio_root / f"{idx // 10000:04d}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                audio_path = shard_dir / f"{idx:08d}.wav"
                sf.write(audio_path, audio, 16000, subtype="PCM_16")
                source_id = str(row.get("id", idx))
                metadata = {
                    "source_id": source_id,
                    "source_index": idx,
                    "source_audio_name": source_name,
                    "hf_id": spec["hf_id"],
                    "revision": spec["revision"],
                    "upstream_filtering": spec["upstream_filtering"],
                }
                if name == "uzbekvoice_filtered":
                    for key in (
                        "original_sentence_id",
                        "sentence_clips_count",
                        "upvotes_count",
                        "downvotes_count",
                        "reported_count",
                        "skipped_clips",
                        "accent_region",
                        "native_language",
                        "gender",
                    ):
                        metadata[key] = row.get(key)
                writer.writerow(
                    {
                        "audio_path": str(audio_path.resolve()),
                        "text": text,
                        "duration": duration,
                        "speaker_id": str(row.get(speaker_column) or "") if speaker_column else "",
                        "split": "train",
                        "source_metadata": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                        "dataset_id": name,
                        "tier": "silver",
                        "trust_weight": 1.5,
                    }
                )
                rows += 1
                hours += duration / 3600
            except Exception as exc:
                rejected += 1
                reject_writer.writerow(
                    {
                        "dataset_id": name,
                        "split": split,
                        "source_index": idx,
                        "reason": repr(exc),
                    }
                )
            if (idx + 1) % 1000 == 0:
                manifest_handle.flush()
                reject_handle.flush()
                progress_path.write_text(
                    json.dumps({"completed_source_rows": idx + 1, "manifest_rows": rows}, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"EXPORT {name}/{split}: source={idx + 1}/{len(data)} "
                    f"written={rows} rejected={rejected}",
                    flush=True,
                )
        progress_path.write_text(
            json.dumps({"completed_source_rows": len(data), "manifest_rows": rows}, indent=2),
            encoding="utf-8",
        )
    return {
        "dataset": name,
        "split": split,
        "source_rows": len(data),
        "manifest_rows": rows,
        "new_hours": hours,
        "export_rejected": rejected,
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pinned SILVER HF snapshots to mono 16 kHz WAV.")
    parser.add_argument("--config", default=str(ROOT / "configs/silver_datasets.yaml"))
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(cfg["output_root"]).expanduser()
    selected = set(args.dataset or cfg["datasets"].keys())
    summary = []
    for name, spec in cfg["datasets"].items():
        if name not in selected:
            continue
        for split in spec["splits"]:
            summary.append(export_split(name, spec, output_root, split))
        summary_path = output_root / name / "processed" / "export_summary.json"
        summary_path.write_text(
            json.dumps([item for item in summary if item["dataset"] == name], indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
