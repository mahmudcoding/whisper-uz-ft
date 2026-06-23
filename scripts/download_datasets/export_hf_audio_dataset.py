#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from text_normalization import normalize_uzbek_text


def row_text(row: dict[str, Any]) -> str:
    for key in ("transcription", "sentence", "text", "raw_transcription"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def row_speaker(row: dict[str, Any]) -> str:
    for key in ("speaker_id", "client_id", "speaker", "gender"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def read_audio_payload(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    if audio.get("bytes"):
        array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        return np.asarray(array, dtype=np.float32), int(sr)
    path = audio.get("path")
    if path:
        array, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(array, dtype=np.float32), int(sr)
    raise RuntimeError("audio row has neither bytes nor path")


def to_mono_16k(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    array, sr = read_audio_payload(audio)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != 16000:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        sr = 16000
    array = np.nan_to_num(array)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array.astype(np.float32), sr


def export_dataset(
    dataset_id: str,
    config: str | None,
    output_dir: Path,
    splits: list[str],
    dataset_name: str,
    tier: str,
) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    manifest_dir = output_dir / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "config": config,
        "dataset_name": dataset_name,
        "tier": tier,
        "splits": {},
    }
    trust_weight = {"gold": 4.0, "silver": 1.5, "bronze": 1.0}[tier]
    fields = ["audio_path", "transcript", "dataset_name", "duration_sec", "speaker_id", "split", "quality_score"]
    canonical_fields = [
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
    for split in splits:
        print(f"LOAD {dataset_id} {config or ''} {split}", flush=True)
        kwargs = {"split": split}
        if config:
            ds = load_dataset(dataset_id, config, **kwargs)
        else:
            ds = load_dataset(dataset_id, **kwargs)
        if "audio" in ds.features:
            ds = ds.cast_column("audio", Audio(decode=False))
        split_audio_dir = audio_dir / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{split}.csv"
        canonical_path = manifest_dir / f"{split}_canonical.csv"
        rows = 0
        total_duration = 0.0
        with manifest_path.open("w", encoding="utf-8", newline="") as f_manifest, canonical_path.open(
            "w", encoding="utf-8", newline=""
        ) as f_canon:
            writer = csv.DictWriter(f_manifest, fieldnames=fields)
            canon_writer = csv.DictWriter(f_canon, fieldnames=canonical_fields)
            writer.writeheader()
            canon_writer.writeheader()
            for idx, row in enumerate(ds):
                audio = row.get("audio")
                if not audio:
                    continue
                array, sr = to_mono_16k(audio)
                if array.size == 0:
                    continue
                duration = float(len(array) / sr)
                if not math.isfinite(duration) or duration <= 0:
                    continue
                transcript = normalize_uzbek_text(row_text(row))
                if not transcript:
                    continue
                wav_path = split_audio_dir / f"{idx:08d}.wav"
                sf.write(str(wav_path), array, sr, subtype="PCM_16")
                speaker_id = row_speaker(row)
                writer.writerow(
                    {
                        "audio_path": str(wav_path),
                        "transcript": transcript,
                        "dataset_name": dataset_name,
                        "duration_sec": duration,
                        "speaker_id": speaker_id,
                        "split": split,
                        "quality_score": 100,
                    }
                )
                canon_writer.writerow(
                    {
                        "audio_path": str(wav_path),
                        "text": transcript,
                        "duration": duration,
                        "speaker_id": speaker_id,
                        "split": split,
                        "source_metadata": json.dumps({"source_index": idx}, ensure_ascii=False),
                        "dataset_id": dataset_name,
                        "tier": tier,
                        "trust_weight": trust_weight,
                    }
                )
                rows += 1
                total_duration += duration
                if rows % 1000 == 0:
                    print(f"{dataset_name}/{split}: exported {rows} rows, {total_duration/3600:.2f}h", flush=True)
        summary["splits"][split] = {
            "rows": rows,
            "hours": total_duration / 3600.0,
            "manifest": str(manifest_path),
            "canonical_manifest": str(canonical_path),
        }
        print(f"DONE {dataset_name}/{split}: {rows} rows, {total_duration/3600:.2f}h", flush=True)
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HF audio dataset splits to 16k mono WAV plus manifests.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tier", default="gold", choices=["gold", "silver", "bronze"])
    parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args()
    export_dataset(
        dataset_id=args.dataset_id,
        config=args.config,
        output_dir=Path(args.output_dir).expanduser(),
        splits=args.splits,
        dataset_name=args.dataset_name,
        tier=args.tier,
    )


if __name__ == "__main__":
    main()
