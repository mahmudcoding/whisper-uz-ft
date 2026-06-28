#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from text_normalization import normalize_uzbek_text


ARCHIVE_ROOT = PurePosixPath("Users/mahmud/FeruzaSpeech")
SPLITS = {
    "train": ("train.tsv", "train"),
    "validation": ("dev.tsv", "val"),
    "test": ("test.tsv", "test"),
}
MANIFEST_FIELDS = [
    "audio_path",
    "transcript",
    "dataset_name",
    "duration_sec",
    "speaker_id",
    "split",
    "quality_score",
]
CANONICAL_FIELDS = [
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
REJECT_FIELDS = [
    "source_audio",
    "split",
    "speaker_id",
    "declared_duration",
    "decoded_duration",
    "reason",
    "transcript",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_name(relative: str) -> str:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {relative!r}")
    return str(ARCHIVE_ROOT / path)


def load_tsv(zf: ZipFile, name: str) -> list[dict[str, str]]:
    with zf.open(str(ARCHIVE_ROOT / name)) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text, delimiter="\t"))


def decode_audio(payload: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        io.BytesIO(payload), dtype="float32", always_2d=False
    )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    return np.nan_to_num(audio), int(sample_rate)


def prepare(zip_path: Path, output_dir: Path, max_duration: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    manifest_root = output_dir / "manifests"
    report_root = output_dir / "reports"
    manifest_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "dataset_id": "k2speech/FeruzaSpeech",
        "dataset_name": "feruzaspeech",
        "tier": "gold",
        "source_archive": str(zip_path),
        "source_archive_sha256": sha256_file(zip_path),
        "license": "other; gated K2Speech terms, academic research/internal use only",
        "max_training_duration_sec": max_duration,
        "splits": {},
    }
    all_rejects: list[dict] = []

    with ZipFile(zip_path) as zf:
        readme_name = str(ARCHIVE_ROOT / "README.md")
        if readme_name in zf.NameToInfo:
            (output_dir / "SOURCE_README.md").write_bytes(zf.read(readme_name))

        for output_name, (tsv_name, split_name) in SPLITS.items():
            rows = load_tsv(zf, tsv_name)
            output_audio = audio_root / output_name
            output_audio.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_root / f"{output_name}.csv"
            canonical_path = manifest_root / f"{output_name}_canonical.csv"
            counters = Counter()
            decoded_hours = 0.0
            kept_hours = 0.0
            speakers: set[str] = set()

            with manifest_path.open(
                "w", encoding="utf-8", newline=""
            ) as manifest_handle, canonical_path.open(
                "w", encoding="utf-8", newline=""
            ) as canonical_handle:
                manifest_writer = csv.DictWriter(
                    manifest_handle, fieldnames=MANIFEST_FIELDS
                )
                canonical_writer = csv.DictWriter(
                    canonical_handle, fieldnames=CANONICAL_FIELDS
                )
                manifest_writer.writeheader()
                canonical_writer.writeheader()

                for index, row in enumerate(rows, start=1):
                    relative_audio = row.get("audio", "").strip()
                    source_name = archive_name(relative_audio)
                    speaker = PurePosixPath(relative_audio).parent.name
                    speaker_id = f"feruzaspeech:{speaker}"
                    transcript = normalize_uzbek_text(row.get("text_latin", ""))
                    declared = float(row.get("duration") or 0.0)
                    reject_reason = ""
                    decoded_duration = 0.0

                    try:
                        payload = zf.read(source_name)
                        audio, sample_rate = decode_audio(payload)
                        decoded_duration = len(audio) / sample_rate
                    except Exception as exc:
                        reject_reason = f"audio_decode_error:{type(exc).__name__}"
                        audio = np.empty(0, dtype=np.float32)
                        sample_rate = 16000

                    if not reject_reason and audio.size == 0:
                        reject_reason = "empty_audio"
                    elif not reject_reason and (
                        not math.isfinite(decoded_duration) or decoded_duration <= 0
                    ):
                        reject_reason = "invalid_duration"
                    elif not reject_reason and decoded_duration > max_duration:
                        reject_reason = "duration_over_30_no_alignment"
                    elif not reject_reason and not transcript:
                        reject_reason = "empty_normalized_transcript"
                    elif not reject_reason and abs(decoded_duration - declared) > 0.1:
                        reject_reason = "declared_duration_mismatch"

                    if decoded_duration > 0:
                        decoded_hours += decoded_duration / 3600.0

                    if reject_reason:
                        counters["rejected"] += 1
                        counters[f"rejected:{reject_reason}"] += 1
                        all_rejects.append(
                            {
                                "source_audio": relative_audio,
                                "split": split_name,
                                "speaker_id": speaker_id,
                                "declared_duration": declared,
                                "decoded_duration": decoded_duration,
                                "reason": reject_reason,
                                "transcript": transcript,
                            }
                        )
                        continue

                    destination = output_audio / speaker / PurePosixPath(relative_audio).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(destination, audio, sample_rate, subtype="PCM_16")
                    written = sf.info(destination)
                    if written.samplerate != 16000 or written.channels != 1:
                        raise RuntimeError(f"Normalization failed for {destination}")

                    metadata = {
                        "source_archive": str(zip_path),
                        "source_audio": relative_audio,
                        "source_tsv": tsv_name,
                        "source_index": index - 1,
                        "text_cyrillic": row.get("text_cyrillic", ""),
                        "words_count": row.get("words_count", ""),
                    }
                    manifest_writer.writerow(
                        {
                            "audio_path": str(destination),
                            "transcript": transcript,
                            "dataset_name": "feruzaspeech",
                            "duration_sec": decoded_duration,
                            "speaker_id": speaker_id,
                            "split": split_name,
                            "quality_score": 100.0,
                        }
                    )
                    canonical_writer.writerow(
                        {
                            "audio_path": str(destination),
                            "text": transcript,
                            "duration": decoded_duration,
                            "speaker_id": speaker_id,
                            "split": split_name,
                            "source_metadata": json.dumps(
                                metadata, ensure_ascii=False, separators=(",", ":")
                            ),
                            "dataset_id": "feruzaspeech",
                            "tier": "gold",
                            "trust_weight": 4.0,
                        }
                    )
                    counters["kept"] += 1
                    kept_hours += decoded_duration / 3600.0
                    speakers.add(speaker_id)
                    if index % 500 == 0:
                        print(
                            f"{output_name}: {index}/{len(rows)} source rows, "
                            f"{counters['kept']} kept, {kept_hours:.2f}h",
                            flush=True,
                        )

            summary["splits"][output_name] = {
                "source_rows": len(rows),
                "kept_rows": counters["kept"],
                "rejected_rows": counters["rejected"],
                "decoded_hours": decoded_hours,
                "kept_hours": kept_hours,
                "speakers": len(speakers),
                "manifest": str(manifest_path),
                "canonical_manifest": str(canonical_path),
                "rejection_reasons": {
                    key.removeprefix("rejected:"): value
                    for key, value in counters.items()
                    if key.startswith("rejected:")
                },
            }
            print(
                f"DONE {output_name}: {counters['kept']} kept, "
                f"{counters['rejected']} rejected, {kept_hours:.2f}h",
                flush=True,
            )

    rejected_path = report_root / "rejected_samples.csv"
    with rejected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REJECT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rejects)

    summary["total_source_rows"] = sum(
        values["source_rows"] for values in summary["splits"].values()
    )
    summary["total_kept_rows"] = sum(
        values["kept_rows"] for values in summary["splits"].values()
    )
    summary["total_rejected_rows"] = len(all_rejects)
    summary["total_decoded_hours"] = sum(
        values["decoded_hours"] for values in summary["splits"].values()
    )
    summary["total_kept_hours"] = sum(
        values["kept_hours"] for values in summary["splits"].values()
    )
    summary["speaker_count"] = sum(
        values["speakers"] for values in summary["splits"].values()
    )
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = [
        "# FeruzaSpeech Preparation Report",
        "",
        f"- Source: `{zip_path}`",
        f"- SHA-256: `{summary['source_archive_sha256']}`",
        f"- Decoded source: {summary['total_decoded_hours']:.4f} hours",
        f"- Training-ready: {summary['total_kept_hours']:.4f} hours",
        f"- Kept clips: {summary['total_kept_rows']:,}",
        f"- Rejected clips: {summary['total_rejected_rows']:,}",
        f"- Speakers: {summary['speaker_count']:,}",
        "",
        "| Split | Source rows | Kept | Rejected | Kept hours | Speakers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in summary["splits"].items():
        report.append(
            f"| {split} | {values['source_rows']:,} | {values['kept_rows']:,} | "
            f"{values['rejected_rows']:,} | {values['kept_hours']:.4f} | "
            f"{values['speakers']:,} |"
        )
    report.extend(
        [
            "",
            "Clips longer than 30 seconds are rejected because the current Whisper "
            "feature pipeline truncates audio at 30 seconds and the archive provides "
            "no word timestamps for lossless transcript-aligned chunking.",
            "",
            "License: gated K2Speech terms restrict use to academic research/internal "
            "use and prohibit redistribution. Review the source terms before any "
            "commercial deployment or artifact publication.",
        ]
    )
    (report_root / "preparation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and export a local FeruzaSpeech ZIP to Gold manifests."
    )
    parser.add_argument(
        "--zip-path", type=Path, default=Path("/home/mahmud/feruzaspeech.zip")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/mahmud/datasets/feruzaspeech"),
    )
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()
    summary = prepare(
        args.zip_path.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.max_duration,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
