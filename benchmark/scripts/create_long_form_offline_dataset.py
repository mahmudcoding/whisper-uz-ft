#!/usr/bin/env python3
"""Create a 5+ hour long-form offline benchmark dataset from USC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
TARGET_SR = 16000


GROUPS = [
    ("short_5s", 5.0, 0.25),
    ("short_10s", 10.0, 0.25),
    ("medium_30s", 30.0, 0.50),
    ("medium_60s", 60.0, 1.00),
    ("long_5_20min", 900.0, 3.00),
]


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return audio


def diverse_source(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            df["source_csv"] = str(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No source CSVs found")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["audio_path"].map(lambda p: Path(str(p)).exists())].copy()
    df = df[df["duration"] >= 1.0].copy()
    if "speaker_id" in df.columns:
        df = df.sample(frac=1.0, random_state=771).sort_values(["speaker_id", "duration"])
    else:
        df = df.sample(frac=1.0, random_state=771)
    return df.reset_index(drop=True)


def choose_rows(df: pd.DataFrame, start_idx: int, target_seconds: float) -> tuple[list[dict], int]:
    rows = []
    seconds = 0.0
    idx = start_idx
    silence_budget = 0.0
    while seconds + silence_budget < target_seconds and len(rows) < len(df):
        row = df.iloc[idx % len(df)].to_dict()
        rows.append(row)
        seconds += float(row["duration"])
        silence_budget += 0.15
        idx += 1
    return rows, idx


def write_recording(rows: list[dict], out_path: Path, target_seconds: float) -> tuple[float, str, str]:
    parts = []
    texts = []
    speaker_ids = []
    silence = np.zeros(int(0.15 * TARGET_SR), dtype=np.float32)
    for row in rows:
        try:
            audio = load_audio(str(row["audio_path"]))
        except Exception:
            continue
        parts.append(audio)
        parts.append(silence)
        texts.append(str(row["text"]))
        if "speaker_id" in row:
            speaker_ids.append(str(row["speaker_id"]))
    if not parts:
        raise RuntimeError(f"No readable source audio for {out_path}")
    audio = np.concatenate(parts)
    max_len = int(target_seconds * TARGET_SR)
    if len(audio) > max_len:
        audio = audio[:max_len]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, TARGET_SR)
    return len(audio) / TARGET_SR, " ".join(texts), "|".join(sorted(set(speaker_ids)))


def build(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.audio_out_dir)
    manifest_path = Path(args.output_csv)
    df = diverse_source([Path(p) for p in args.source_csvs])
    records = []
    source_idx = 0

    for group_name, recording_seconds, target_hours in GROUPS:
        target_total = target_hours * 3600.0
        made = 0.0
        n = 0
        while made < target_total:
            target = recording_seconds
            if group_name == "long_5_20min":
                # Cycle through 5, 10, 15, and 20 minute recordings.
                target = [300.0, 600.0, 900.0, 1200.0][n % 4]
            rows, source_idx = choose_rows(df, source_idx, target)
            out_path = out_dir / group_name / f"{group_name}_{n:04d}.wav"
            duration, text, speaker_ids = write_recording(rows, out_path, target)
            records.append(
                {
                    "audio_path": str(out_path.resolve()),
                    "text": text,
                    "duration": duration,
                    "duration_group": group_name,
                    "speaker_id": speaker_ids,
                    "split": "long_form_offline_benchmark",
                    "source_metadata": json.dumps(
                        {
                            "target_duration_seconds": target,
                            "source_samples": len(rows),
                            "source_csvs": sorted(set(str(r.get("source_csv", "")) for r in rows)),
                        }
                    ),
                }
            )
            made += duration
            n += 1
    manifest = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    summary = manifest.groupby("duration_group")["duration"].agg(["count", "sum", "mean"])
    print(summary.to_string())
    print(f"total_hours={manifest['duration'].sum() / 3600.0:.3f}")
    print(f"manifest={manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csvs", nargs="+", default=["data/test.csv", "data/val.csv", "data/train.csv"])
    p.add_argument("--audio-out-dir", default="benchmark/datasets/long_form_offline_audio")
    p.add_argument("--output-csv", default="benchmark/datasets/long_form_offline_5h.csv")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
