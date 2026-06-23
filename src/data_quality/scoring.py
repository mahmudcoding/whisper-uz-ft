from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from filtering.scoring import score_sample
from text_normalization import normalize_uzbek_text


def audio_stats(path: str, silence_threshold_db: float = -40.0) -> dict:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return {"audio_error": "empty_audio"}
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)))
    db = 20.0 * math.log10(rms + 1e-12)
    frame = max(1, int(sr * 0.02))
    usable = len(audio) - (len(audio) % frame)
    if usable <= 0:
        silence_pct = 1.0
    else:
        frames = audio[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        frame_db = 20.0 * np.log10(frame_rms + 1e-12)
        silence_pct = float(np.mean(frame_db < silence_threshold_db))
    snr_proxy = float(max(0.0, min(60.0, db - silence_threshold_db)))
    return {
        "rms_db": db,
        "peak": peak,
        "silence_pct": silence_pct,
        "snr_proxy_db": snr_proxy,
    }


def classify(score: float, tier_hint: str) -> str:
    if score < 55:
        return "reject"
    if score < 75:
        return "bronze"
    if tier_hint == "gold" and score >= 85:
        return "gold"
    if score >= 75:
        return "silver"
    return "bronze"


def score_manifest(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as f_in, output_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = list(reader.fieldnames or [])
        extra = [
            "normalized_text",
            "quality_score",
            "quality_class",
            "quality_decision",
            "quality_reasons",
            "chars_per_second",
            "transcript_chars",
            "rms_db",
            "peak",
            "silence_pct",
            "snr_proxy_db",
        ]
        writer = csv.DictWriter(f_out, fieldnames=list(dict.fromkeys(fields + extra)))
        writer.writeheader()
        for row in reader:
            text = row.get("text", "")
            duration = float(row.get("duration") or 0)
            q = score_sample(
                text,
                duration,
                asr_similarity=float(row["asr_similarity"]) if row.get("asr_similarity") else None,
                asr_cer=float(row["asr_cer"]) if row.get("asr_cer") else None,
                asr_wer=float(row["asr_wer"]) if row.get("asr_wer") else None,
            )
            score = q.score
            reasons = list(q.reasons)
            try:
                stats = audio_stats(row.get("audio_path", ""))
                silence_pct = float(stats.get("silence_pct", 0))
                snr_proxy = float(stats.get("snr_proxy_db", 0))
                if silence_pct > 0.65:
                    score -= 15
                    reasons.append("high_silence")
                if snr_proxy < 10:
                    score -= 15
                    reasons.append("low_snr_proxy")
            except Exception as exc:
                stats = {"audio_error": repr(exc)}
                score -= 25
                reasons.append("audio_read_error")
            score = max(0.0, min(100.0, score))
            row.update(stats)
            row["normalized_text"] = normalize_uzbek_text(text)
            row["quality_score"] = score
            row["quality_class"] = classify(score, row.get("tier", ""))
            row["quality_decision"] = "reject" if score < 55 else "keep"
            row["quality_reasons"] = "|".join(sorted(set(reasons)))
            row["chars_per_second"] = q.chars_per_second
            row["transcript_chars"] = q.transcript_chars
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    score_manifest(Path(args.input_csv), Path(args.output_csv))


if __name__ == "__main__":
    main()

