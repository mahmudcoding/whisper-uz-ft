from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


def pcm_sha1(path: str | Path, seconds: float | None = None) -> str:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if seconds is not None:
        audio = audio[: int(sr * seconds)]
    audio = np.nan_to_num(audio)
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype("<i2", copy=False)
    h = hashlib.sha1()
    h.update(str(sr).encode("ascii"))
    h.update(audio_i16.tobytes())
    return h.hexdigest()


def duration_bucket(duration: float, resolution: float = 0.1) -> int:
    return int(round(float(duration) / resolution))


def add_audio_hashes(input_csv: Path, output_csv: Path, sample_seconds: float | None = None) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as f_in, output_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = list(reader.fieldnames or [])
        for field in ["audio_sha1", "audio_head_sha1", "duration_bucket_100ms", "hash_error"]:
            if field not in fields:
                fields.append(field)
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            path = row.get("audio_path", "")
            try:
                row["audio_sha1"] = pcm_sha1(path)
                row["audio_head_sha1"] = pcm_sha1(path, seconds=sample_seconds or 10.0)
            except Exception as exc:
                row["audio_sha1"] = ""
                row["audio_head_sha1"] = ""
                row["hash_error"] = repr(exc)
            else:
                row["hash_error"] = ""
            try:
                row["duration_bucket_100ms"] = duration_bucket(float(row.get("duration") or 0))
            except Exception:
                row["duration_bucket_100ms"] = ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sample-seconds", type=float, default=10.0)
    args = parser.parse_args()
    add_audio_hashes(Path(args.input_csv), Path(args.output_csv), args.sample_seconds)


if __name__ == "__main__":
    main()
