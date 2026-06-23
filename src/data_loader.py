from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from normalize import normalize_uzbek_text

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
TEXT_COLUMNS = ("text", "transcript", "sentence", "transcription")
AUDIO_COLUMNS = (
    "audio",
    "audio_path",
    "audio_filepath",
    "path",
    "file",
    "file_path",
    "filepath",
    "filename",
    "wav",
    "flac",
    "mp3",
)
SPEAKER_COLUMNS = ("speaker", "speaker_id", "speakerid", "client_id", "clientid", "utt_spk")
OFFICIAL_TRAIN_NAMES = {"train", "training"}
OFFICIAL_VAL_NAMES = {"dev", "val", "valid", "validation"}
OFFICIAL_TEST_NAMES = {"test", "testing"}


def canonical_split_name(name: str) -> str | None:
    low = name.lower()
    if low in OFFICIAL_TRAIN_NAMES:
        return "train"
    if low in OFFICIAL_VAL_NAMES:
        return "val"
    if low in OFFICIAL_TEST_NAMES:
        return "test"
    return None


def infer_split_from_path(path: Path, dataset_dir: Path) -> str | None:
    try:
        rel_parts = path.relative_to(dataset_dir).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts:
        split = canonical_split_name(part)
        if split:
            return split
    return None


def infer_speaker_id(audio: Path) -> str:
    stem = audio.stem
    if "_data_" in stem:
        return stem.split("_data_", 1)[0]
    return stem.split("_", 1)[0]


def _read_json(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)


def find_metadata_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
    )


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for c in columns:
        low = c.lower()
        if any(candidate in low for candidate in candidates):
            return c
    return None


def read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return _read_json(path)


def _resolve_audio_path(value: object, base_dir: Path, dataset_dir: Path | None = None) -> str:
    if isinstance(value, dict):
        for key in ("path", "audio_path", "file", "filename"):
            if key in value and value[key]:
                value = value[key]
                break
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    direct = base_dir / path
    if direct.exists():
        return str(direct.resolve())
    root = dataset_dir or base_dir
    root_direct = root / path
    if root_direct.exists():
        return str(root_direct.resolve())
    matches = list(root.rglob(path.name))
    if matches:
        return str(matches[0].resolve())
    return str(direct.resolve())


def load_from_metadata(dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for meta in find_metadata_files(dataset_dir):
        frame = read_metadata(meta)
        if frame.empty:
            continue
        audio_col = detect_column(frame.columns, AUDIO_COLUMNS)
        text_col = detect_column(frame.columns, TEXT_COLUMNS)
        if not audio_col or not text_col:
            continue
        speaker_col = detect_column(frame.columns, SPEAKER_COLUMNS)
        out = pd.DataFrame(
            {
                "audio_path": frame[audio_col].map(lambda v: _resolve_audio_path(v, meta.parent, dataset_dir)),
                "text": frame[text_col],
                "source_metadata": str(meta),
            }
        )
        if speaker_col:
            out["speaker_id"] = frame[speaker_col].astype(str)
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_from_audio_folders(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for audio in sorted(p for p in dataset_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS):
        transcript = None
        for txt in (
            audio.with_suffix(".txt"),
            audio.with_suffix(".lab"),
            audio.parent / f"{audio.stem}.trans.txt",
            audio.parent / f"{audio.stem}.normalized.txt",
        ):
            if txt.exists():
                transcript = txt.read_text(encoding="utf-8", errors="ignore").strip()
                break
        if transcript:
            rows.append(
                {
                    "audio_path": str(audio.resolve()),
                    "text": transcript,
                    "speaker_id": infer_speaker_id(audio),
                    "split": infer_split_from_path(audio, dataset_dir),
                    "source_metadata": "sidecar",
                }
            )
    return pd.DataFrame(rows)


def load_hf_dataset(name: str, split: str, cache_dir: Path | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    frame = ds.to_pandas()
    audio_col = detect_column(frame.columns, AUDIO_COLUMNS)
    text_col = detect_column(frame.columns, TEXT_COLUMNS)
    if not audio_col or not text_col:
        raise ValueError(f"Could not detect audio/text columns in HuggingFace dataset columns: {list(frame.columns)}")
    speaker_col = detect_column(frame.columns, SPEAKER_COLUMNS)
    out = pd.DataFrame({"audio_path": frame[audio_col], "text": frame[text_col], "source_metadata": name})
    if speaker_col:
        out["speaker_id"] = frame[speaker_col].astype(str)
    return out


def load_dataset_flexible(dataset_dir: Path, hf_dataset: str | None = None, hf_split: str = "train") -> pd.DataFrame:
    if hf_dataset:
        return load_hf_dataset(hf_dataset, hf_split, dataset_dir / ".hf_cache")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    frame = load_from_metadata(dataset_dir)
    if frame.empty:
        frame = load_from_audio_folders(dataset_dir)
    if frame.empty:
        raise ValueError(
            f"No usable dataset found in {dataset_dir}. Expected CSV/JSON metadata with audio and text columns, "
            "or audio files with .txt/.lab sidecar transcripts."
        )
    return frame


def audio_duration_seconds(path: str) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa

        return float(librosa.get_duration(path=path))
    except Exception:
        return None


def inspect_loaded_dataset(frame: pd.DataFrame, n: int = 5) -> dict:
    samples = []
    for row in frame.head(n).to_dict("records"):
        samples.append(
            {
                "audio_path": row.get("audio_path"),
                "audio_exists": bool(Path(str(row.get("audio_path", ""))).exists()),
                "text": normalize_uzbek_text(row.get("text", "")),
                "speaker_id": row.get("speaker_id"),
                "source_metadata": row.get("source_metadata"),
            }
        )
    return {
        "sample_count": int(len(frame)),
        "columns": list(frame.columns),
        "samples": samples,
        "transcript_examples": [s["text"] for s in samples],
        "resolved_audio_paths": [s["audio_path"] for s in samples],
    }


def clean_dataset(
    frame: pd.DataFrame,
    lowercase: bool = False,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    min_chars: int = 2,
    max_chars: int = 500,
) -> tuple[pd.DataFrame, dict]:
    reasons: Counter[str] = Counter()
    rows = []
    seen = set()
    total = len(frame)
    for row in frame.to_dict("records"):
        audio_path = str(row.get("audio_path", "")).strip()
        text = normalize_uzbek_text(row.get("text", ""), lowercase=lowercase)
        key = (audio_path, text)
        if not text:
            reasons["empty_transcript"] += 1
            continue
        if len(text) < min_chars:
            reasons["transcript_too_short"] += 1
            continue
        if len(text) > max_chars:
            reasons["transcript_too_long"] += 1
            continue
        if not audio_path or not Path(audio_path).exists():
            reasons["missing_audio"] += 1
            continue
        duration = audio_duration_seconds(audio_path)
        if duration is None or math.isnan(duration) or duration <= 0:
            reasons["corrupted_audio"] += 1
            continue
        if duration < min_duration:
            reasons["duration_too_short"] += 1
            continue
        if duration > max_duration:
            reasons["duration_too_long"] += 1
            continue
        if key in seen:
            reasons["duplicate_sample"] += 1
            continue
        seen.add(key)
        out = dict(row)
        out["audio_path"] = audio_path
        out["text"] = text
        out["duration"] = duration
        rows.append(out)
    clean = pd.DataFrame(rows)
    report = {
        "total_samples": total,
        "final_samples": int(len(clean)),
        "removed_samples": int(total - len(clean)),
        "reasons": dict(reasons),
        "final_hours": float(clean["duration"].sum() / 3600.0) if not clean.empty else 0.0,
    }
    return clean, report


def split_dataset(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if frame.empty:
        raise ValueError("Cannot split empty dataset after cleaning.")
    if "split" in frame.columns:
        split_values = set(frame["split"].dropna().astype(str))
        if {"train", "val", "test"}.issubset(split_values):
            train = frame[frame["split"] == "train"]
            val = frame[frame["split"] == "val"]
            test = frame[frame["split"] == "test"]
            if not train.empty and not val.empty and not test.empty:
                return train, val, test, "official_split"
    if "speaker_id" in frame.columns and frame["speaker_id"].nunique() >= 20:
        speakers = frame[["speaker_id"]].drop_duplicates()
        train_spk, tmp_spk = train_test_split(speakers, test_size=0.10, random_state=seed)
        val_spk, test_spk = train_test_split(tmp_spk, test_size=0.50, random_state=seed)
        train = frame[frame["speaker_id"].isin(train_spk["speaker_id"])]
        val = frame[frame["speaker_id"].isin(val_spk["speaker_id"])]
        test = frame[frame["speaker_id"].isin(test_spk["speaker_id"])]
        return train, val, test, "speaker_independent"
    bins = min(10, max(2, len(frame) // 100))
    labels = pd.qcut(frame["duration"], q=bins, duplicates="drop", labels=False)
    stratify = labels if labels.value_counts().min() >= 2 else None
    train, tmp = train_test_split(frame, test_size=0.10, random_state=seed, stratify=stratify)
    tmp_labels = pd.qcut(tmp["duration"], q=min(5, max(2, len(tmp) // 20)), duplicates="drop", labels=False)
    tmp_strat = tmp_labels if tmp_labels.value_counts().min() >= 2 else None
    val, test = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=tmp_strat)
    return train, val, test, "duration_stratified_random"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, normalize, clean, and split Uzbek ASR data.")
    parser.add_argument("--dataset-dir", default=os.path.expanduser("~/datasets/usc"))
    parser.add_argument("--out-dir", default=os.path.expanduser("~/whisper-uz-ft/data"))
    parser.add_argument("--report", default=os.path.expanduser("~/whisper-uz-ft/logs/data_cleaning_report.json"))
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).expanduser().parent.mkdir(parents=True, exist_ok=True)

    raw = load_dataset_flexible(dataset_dir, args.hf_dataset, args.hf_split)
    if args.inspect_only:
        print(json.dumps(inspect_loaded_dataset(raw), indent=2, ensure_ascii=False))
        return
    clean, report = clean_dataset(raw, lowercase=args.lowercase)
    train, val, test, strategy = split_dataset(clean, seed=args.seed)
    report["split_strategy"] = strategy
    report["splits"] = {"train": len(train), "val": len(val), "test": len(test)}
    report["split_hours"] = {
        "train": float(train["duration"].sum() / 3600.0),
        "val": float(val["duration"].sum() / 3600.0),
        "test": float(test["duration"].sum() / 3600.0),
    }

    keep_cols = [c for c in ("audio_path", "text", "duration", "speaker_id", "split", "source_metadata") if c in clean.columns]
    train[keep_cols].to_csv(out_dir / "train.csv", index=False)
    val[keep_cols].to_csv(out_dir / "val.csv", index=False)
    test[keep_cols].to_csv(out_dir / "test.csv", index=False)
    Path(args.report).expanduser().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
