#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from filtering.scoring import score_sample
from text_normalization import normalize_uzbek_text


ROOT = Path(__file__).resolve().parents[2]


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def pcm_hash(audio: np.ndarray, sr: int) -> str:
    pcm = (np.clip(np.nan_to_num(audio), -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    digest = hashlib.sha1()
    digest.update(str(sr).encode("ascii"))
    digest.update(pcm.tobytes())
    return digest.hexdigest()


def analyze_audio(path: str) -> dict[str, Any]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000 or channels != 1:
        raise ValueError(f"noncanonical audio sr={sr} channels={channels}")
    if not audio.size:
        raise ValueError("empty audio")
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    rms_db = 20.0 * math.log10(rms + 1e-12)
    peak = float(np.max(np.abs(audio)))
    frame = max(1, int(sr * 0.02))
    usable = len(audio) - len(audio) % frame
    if usable:
        frames = audio[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        frame_db = 20.0 * np.log10(frame_rms + 1e-12)
        silence = float(np.mean(frame_db < -40.0))
    else:
        silence = 1.0
    return {
        "sample_rate": sr,
        "channels": channels,
        "audio_sha1": pcm_hash(audio, sr),
        "audio_head_sha1": pcm_hash(audio[: sr * 10], sr),
        "rms_db": rms_db,
        "peak": peak,
        "silence_pct": silence,
        "snr_proxy_db": max(0.0, min(60.0, rms_db + 40.0)),
    }


def gold_indexes(gold_quality: Path) -> dict[str, Any]:
    indexes = {
        "audio": {},
        "head_duration": {},
        "text_duration": {},
        "locked_text": {},
    }
    with gold_quality.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            if row.get("audio_sha1"):
                indexes["audio"][row["audio_sha1"]] = split
            if row.get("audio_head_sha1"):
                indexes["head_duration"][
                    (row["audio_head_sha1"], row.get("duration_bucket_100ms", ""))
                ] = split
            if row.get("transcript_sha1"):
                indexes["text_duration"][
                    (row["transcript_sha1"], row.get("duration_bucket_100ms", ""))
                ] = split
                if split in {"val", "validation", "test"}:
                    indexes["locked_text"][row["transcript_sha1"]] = split
    return indexes


def manifest_paths(cfg: dict[str, Any]) -> list[Path]:
    output_root = Path(cfg["output_root"]).expanduser()
    paths = []
    for name, spec in cfg["datasets"].items():
        for split in spec["splits"]:
            paths.append(output_root / name / "processed/manifests" / f"{split}_canonical.csv")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict heuristic filtering and Gold dedup for SILVER.")
    parser.add_argument("--config", default=str(ROOT / "configs/silver_datasets.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thresholds = cfg["thresholds"]
    work_dir = Path(cfg["work_dir"]).expanduser()
    report_dir = Path(cfg["report_dir"]).expanduser()
    dedup_dir = Path(cfg["dedup_report_dir"]).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)

    gold = gold_indexes(ROOT / "data/gold_work/gold_quality.csv")
    output_path = work_dir / "silver_prefilter_scored.csv"
    candidate_path = work_dir / "silver_teacher_candidates.csv"
    rejected_path = report_dir / "prefilter_rejected.csv"
    fields = [
        "audio_path", "text", "duration", "speaker_id", "split", "source_metadata",
        "dataset_id", "tier", "trust_weight", "normalized_text", "transcript_sha1",
        "duration_bucket_100ms", "audio_sha1", "audio_head_sha1", "rms_db", "peak",
        "silence_pct", "snr_proxy_db", "chars_per_second", "heuristic_quality_score",
        "prefilter_decision", "rejection_reasons", "gold_overlap_split",
    ]
    seen_audio: set[str] = set()
    seen_head_duration: set[tuple[str, str]] = set()
    seen_text_duration: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    hours: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8", newline="") as out_handle, candidate_path.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_handle, rejected_path.open("w", encoding="utf-8", newline="") as rejected_handle:
        out_writer = csv.DictWriter(out_handle, fieldnames=fields)
        candidate_writer = csv.DictWriter(candidate_handle, fieldnames=fields)
        rejected_writer = csv.DictWriter(rejected_handle, fieldnames=fields)
        out_writer.writeheader()
        candidate_writer.writeheader()
        rejected_writer.writeheader()

        processed = 0
        for path in manifest_paths(cfg):
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    processed += 1
                    reasons: list[str] = []
                    dataset_id = row["dataset_id"]
                    duration = float(row.get("duration") or 0)
                    normalized = normalize_uzbek_text(row.get("text"))
                    transcript_sha1 = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
                    bucket = str(int(round(duration / 0.1))) if duration > 0 else ""
                    q = score_sample(normalized, duration)
                    score = float(q.score)
                    reasons.extend(q.reasons)
                    try:
                        stats = analyze_audio(row["audio_path"])
                    except Exception as exc:
                        stats = {
                            "audio_sha1": "", "audio_head_sha1": "", "rms_db": "",
                            "peak": "", "silence_pct": 1.0, "snr_proxy_db": 0.0,
                        }
                        reasons.append(f"audio_validation_error:{type(exc).__name__}")
                        score = 0.0

                    cps = len(normalized) / duration if duration > 0 else 0.0
                    if duration < float(thresholds["min_duration_sec"]):
                        reasons.append("duration_too_short")
                    if duration > float(thresholds["max_duration_sec"]):
                        reasons.append("duration_too_long")
                    if cps < float(thresholds["min_chars_per_second"]):
                        reasons.append("reading_speed_too_low")
                    if cps > float(thresholds["max_chars_per_second"]):
                        reasons.append("reading_speed_too_high")
                    if float(stats["silence_pct"]) > float(thresholds["max_silence_fraction"]):
                        reasons.append("excessive_silence")
                        score -= 20
                    if float(stats["snr_proxy_db"]) < float(thresholds["min_snr_proxy_db"]):
                        reasons.append("low_snr_proxy")
                        score -= 15

                    metadata = json.loads(row.get("source_metadata") or "{}")
                    if dataset_id == "uzbekvoice_filtered":
                        if int(metadata.get("reported_count") or 0) > 0:
                            reasons.append("upstream_reported")
                        if int(metadata.get("downvotes_count") or 0) > int(metadata.get("upvotes_count") or 0):
                            reasons.append("upstream_votes_negative")

                    audio_hash = str(stats["audio_sha1"])
                    head_key = (str(stats["audio_head_sha1"]), bucket)
                    text_key = (transcript_sha1, bucket)
                    gold_split = ""
                    if audio_hash and audio_hash in gold["audio"]:
                        gold_split = gold["audio"][audio_hash]
                        reasons.append("exact_audio_overlap_gold")
                    elif head_key[0] and head_key in gold["head_duration"]:
                        gold_split = gold["head_duration"][head_key]
                        reasons.append("near_audio_overlap_gold")
                    elif text_key in gold["text_duration"]:
                        gold_split = gold["text_duration"][text_key]
                        reasons.append("text_duration_overlap_gold")
                    elif transcript_sha1 in gold["locked_text"]:
                        gold_split = gold["locked_text"][transcript_sha1]
                        reasons.append("transcript_overlap_locked_eval")

                    if audio_hash and audio_hash in seen_audio:
                        reasons.append("exact_audio_duplicate_silver")
                    elif head_key[0] and head_key in seen_head_duration:
                        reasons.append("near_audio_duplicate_silver")
                    elif text_key in seen_text_duration:
                        reasons.append("text_duration_duplicate_silver")
                    seen_audio.add(audio_hash)
                    seen_head_duration.add(head_key)
                    seen_text_duration.add(text_key)

                    hard_reasons = {
                        "duration_too_short", "duration_too_long", "reading_speed_too_low",
                        "reading_speed_too_high", "excessive_silence", "low_snr_proxy",
                        "upstream_reported", "upstream_votes_negative",
                        "exact_audio_overlap_gold", "near_audio_overlap_gold",
                        "text_duration_overlap_gold", "transcript_overlap_locked_eval",
                        "exact_audio_duplicate_silver", "near_audio_duplicate_silver",
                        "text_duration_duplicate_silver",
                    }
                    hard_reject = any(reason.split(":", 1)[0] in hard_reasons for reason in reasons)
                    score = max(0.0, min(100.0, score))
                    decision = "reject" if hard_reject or score < 70 else "teacher_candidate"
                    result = {
                        **{key: row.get(key, "") for key in FIELDS_BASE},
                        "normalized_text": normalized,
                        "transcript_sha1": transcript_sha1,
                        "duration_bucket_100ms": bucket,
                        "audio_sha1": audio_hash,
                        "audio_head_sha1": stats["audio_head_sha1"],
                        "rms_db": stats["rms_db"],
                        "peak": stats["peak"],
                        "silence_pct": stats["silence_pct"],
                        "snr_proxy_db": stats["snr_proxy_db"],
                        "chars_per_second": cps,
                        "heuristic_quality_score": score,
                        "prefilter_decision": decision,
                        "rejection_reasons": "|".join(sorted(set(reasons))),
                        "gold_overlap_split": gold_split,
                    }
                    out_writer.writerow(result)
                    counts[f"{dataset_id}:{decision}"] += 1
                    hours[f"{dataset_id}:{decision}"] += duration / 3600
                    if decision == "teacher_candidate":
                        candidate_writer.writerow(result)
                    else:
                        rejected_writer.writerow(result)
                    if processed % 1000 == 0:
                        print(f"PREFILTER processed={processed} candidates={sum(v for k,v in counts.items() if k.endswith(':teacher_candidate'))}", flush=True)

    summary = {
        "counts": dict(counts),
        "hours": dict(hours),
        "thresholds": thresholds,
        "candidate_manifest": str(candidate_path),
        "rejected_manifest": str(rejected_path),
    }
    (report_dir / "prefilter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


FIELDS_BASE = [
    "audio_path", "text", "duration", "speaker_id", "split", "source_metadata",
    "dataset_id", "tier", "trust_weight",
]


if __name__ == "__main__":
    main()
