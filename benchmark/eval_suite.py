#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import soundfile as sf
import librosa
from jiwer import cer, wer
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]


def normalize_metric_text(text: str) -> str:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from text_normalization import normalize_uzbek_text

    return normalize_uzbek_text(text)


def _torch_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def load_audio_for_pipeline(audio_path: str, target_sr: int = 16000) -> dict:
    path = Path(audio_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    array, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return {"array": array, "sampling_rate": target_sr}


def evaluate_transformers(
    model_path: str,
    dataset_csv: Path,
    max_samples: int | None,
    language: str,
    batch_size: int,
    precision: str,
) -> dict:
    df = pd.read_csv(dataset_csv)
    if max_samples:
        df = df.head(max_samples)
    device = 0 if torch.cuda.is_available() else -1
    dtype = _torch_dtype(precision) if torch.cuda.is_available() else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        torch_dtype=dtype,
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    refs: list[str] = []
    hyps: list[str] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    records = df.to_dict("records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        inputs = [load_audio_for_pipeline(str(row["audio_path"])) for row in batch]
        outputs = asr(inputs, batch_size=batch_size)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for row, out in zip(batch, outputs):
            hyp = out["text"] if isinstance(out, dict) else str(out)
            refs.append(str(row["text"]))
            hyps.append(hyp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0
    audio_seconds = float(df["duration"].sum()) if "duration" in df else 0.0
    peak_vram_mib = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    nrefs = [normalize_metric_text(x) for x in refs]
    nhyps = [normalize_metric_text(x) for x in hyps]
    return {
        "model_path": model_path,
        "dataset": str(dataset_csv),
        "precision": precision,
        "batch_size": batch_size,
        "samples": int(len(df)),
        "audio_seconds": audio_seconds,
        "audio_hours": audio_seconds / 3600.0,
        "runtime_seconds": runtime,
        "rtf": runtime / audio_seconds if audio_seconds else None,
        "audio_hours_per_hour": (audio_seconds / runtime) if runtime else None,
        "peak_vram_mib": peak_vram_mib,
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
        "normalized_wer": float(wer(nrefs, nhyps)),
        "normalized_cer": float(cer(nrefs, nhyps)),
        "examples": [{"reference": r, "prediction": h} for r, h in list(zip(refs, hyps))[:20]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Whisper evaluation suite.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", "--manifest", dest="dataset", default=str(ROOT / "data/test.csv"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--language", choices=["uz"], default="uz")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--output", default=str(ROOT / "benchmark/reports/eval_suite_result.json"))
    args = parser.parse_args()
    result = evaluate_transformers(
        args.model_path,
        Path(args.dataset),
        args.max_samples,
        args.language,
        max(1, args.batch_size),
        args.precision,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
