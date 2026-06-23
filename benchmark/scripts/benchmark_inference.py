#!/usr/bin/env python3
"""Production inference benchmark runner for Whisper models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
import torch
from jiwer import cer, wer


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "benchmark" / "datasets"
RESULTS_DIR = ROOT / "benchmark" / "results"
DEFAULT_GPU_CSV = RESULTS_DIR / "gpu_metrics.csv"


@dataclass
class Sample:
    audio_path: str
    text: str
    duration: float
    segments: list[str] | None = None


class TelemetrySampler:
    def __init__(self, output_csv: Path, interval: float = 1.0) -> None:
        self.output_csv = output_csv
        self.interval = interval
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)
        self._write()

    def _query_gpu(self) -> dict[str, float | None]:
        data: dict[str, float | None] = {
            "gpu_util_pct": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
        }
        if not self._has_nvidia_smi:
            return data
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2)
            first = out.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            keys = ["gpu_util_pct", "vram_used_mb", "vram_total_mb", "gpu_temp_c", "gpu_power_w"]
            for key, value in zip(keys, parts):
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = None
        except Exception:
            pass
        return data

    def _loop(self) -> None:
        while not self._stop.is_set():
            vm = psutil.virtual_memory()
            row: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_mb": vm.used / (1024 * 1024),
                "ram_total_mb": vm.total / (1024 * 1024),
            }
            row.update(self._query_gpu())
            self.rows.append(row)
            self._stop.wait(self.interval)

    def _write(self) -> None:
        fields = [
            "timestamp_utc",
            "gpu_util_pct",
            "vram_used_mb",
            "vram_total_mb",
            "gpu_temp_c",
            "gpu_power_w",
            "cpu_percent",
            "ram_used_mb",
            "ram_total_mb",
        ]
        with self.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def summary(self) -> dict[str, float | None]:
        def avg(key: str) -> float | None:
            vals = [float(r[key]) for r in self.rows if r.get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        def peak(key: str) -> float | None:
            vals = [float(r[key]) for r in self.rows if r.get(key) is not None]
            return float(max(vals)) if vals else None

        return {
            "avg_gpu_util_pct": avg("gpu_util_pct"),
            "peak_gpu_util_pct": peak("gpu_util_pct"),
            "avg_vram_mb": avg("vram_used_mb"),
            "peak_vram_mb": peak("vram_used_mb"),
            "avg_gpu_temp_c": avg("gpu_temp_c"),
            "peak_gpu_temp_c": peak("gpu_temp_c"),
            "avg_gpu_power_w": avg("gpu_power_w"),
            "peak_gpu_power_w": peak("gpu_power_w"),
            "avg_cpu_percent": avg("cpu_percent"),
            "peak_cpu_percent": peak("cpu_percent"),
            "avg_ram_mb": avg("ram_used_mb"),
            "peak_ram_mb": peak("ram_used_mb"),
        }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    idx = (len(arr) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(arr[lo])
    return float(arr[lo] * (hi - idx) + arr[hi] * (idx - lo))


def read_duration(path: str) -> float:
    try:
        info = sf.info(path)
        return float(info.frames / info.samplerate)
    except Exception:
        return 0.0


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    try:
        import librosa

        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        return audio.astype(np.float32), int(sr)
    except Exception:
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio.astype(np.float32), int(sr)


def resolve_dataset(dataset: str) -> Path:
    candidate = Path(dataset).expanduser()
    if candidate.exists():
        return candidate.resolve()
    named = DATASET_DIR / f"{dataset}.csv"
    if named.exists():
        return named.resolve()
    raise FileNotFoundError(f"Dataset not found: {dataset} (looked for {named})")


def load_samples(path: Path, max_samples: int | None) -> list[Sample]:
    df = pd.read_csv(path)
    if max_samples:
        df = df.head(max_samples)
    text_col = next((c for c in ["text", "transcript", "sentence", "transcription"] if c in df.columns), None)
    samples: list[Sample] = []
    for _, row in df.iterrows():
        segments = None
        if "segments" in row and isinstance(row["segments"], str) and row["segments"]:
            segments = [p for p in row["segments"].split("|") if p]
        audio_path = str(row.get("audio_path", segments[0] if segments else ""))
        if not audio_path:
            continue
        duration = float(row.get("duration", 0.0) or 0.0)
        if duration <= 0 and segments:
            duration = sum(read_duration(p) for p in segments)
        elif duration <= 0:
            duration = read_duration(audio_path)
        samples.append(Sample(audio_path=audio_path, text=str(row.get(text_col, "")) if text_col else "", duration=duration, segments=segments))
    return samples


def load_faster_whisper(model_path: str, precision: str):
    from faster_whisper import WhisperModel

    compute_type = {
        "fp16": "float16",
        "float16": "float16",
        "int8": "int8",
        "int8_float16": "int8_float16",
        "float32": "float32",
    }[precision]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    pipeline = None
    try:
        from faster_whisper import BatchedInferencePipeline

        pipeline = BatchedInferencePipeline(model=model)
    except Exception:
        pipeline = None
    return {"model": model, "pipeline": pipeline}


def transcribe_faster(bundle: Any, path: str, beam_size: int, batch_size: int, vad: bool, language: str, task: str) -> str:
    model = bundle["pipeline"] if isinstance(bundle, dict) and batch_size > 1 and bundle.get("pipeline") is not None else bundle["model"]
    kwargs = {"beam_size": beam_size, "vad_filter": vad, "language": language, "task": task}
    if batch_size > 1 and isinstance(bundle, dict) and bundle.get("pipeline") is not None:
        kwargs["batch_size"] = batch_size
        if not vad:
            duration = read_duration(path)
            chunk = 30.0
            kwargs["clip_timestamps"] = [
                {"start": start, "end": min(start + chunk, duration)}
                for start in np.arange(0.0, max(duration, 0.001), chunk)
            ]
    segments, _info = model.transcribe(path, **kwargs)
    return " ".join(seg.text.strip() for seg in segments).strip()


def load_transformers(model_path: str, precision: str):
    if precision in {"int8", "int8_float16"}:
        raise ValueError("Transformers int8 is intentionally not used here; use faster-whisper/CTranslate2 for int8.")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" and precision in {"fp16", "float16"} else torch.float32
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model, device, dtype


def transcribe_transformers(bundle: Any, path: str, beam_size: int, language: str, task: str) -> str:
    processor, model, device, dtype = bundle
    audio, sr = load_audio(path, 16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    with torch.inference_mode():
        ids = model.generate(input_features, attention_mask=attention_mask, forced_decoder_ids=forced_ids, num_beams=beam_size)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe_path(engine: str, bundle: Any, path: str, args: argparse.Namespace) -> str:
    if engine in {"faster-whisper", "ctranslate2"}:
        return transcribe_faster(bundle, path, args.beam_size, args.batch_size, args.vad == "on", args.language, args.task)
    if engine == "transformers":
        return transcribe_transformers(bundle, path, args.beam_size, args.language, args.task)
    raise ValueError(engine)


def split_to_temp_wavs(path: str, chunk_seconds: float, tmpdir: Path) -> list[tuple[str, float]]:
    audio, sr = load_audio(path, 16000)
    chunk_len = max(1, int(chunk_seconds * sr))
    chunks: list[tuple[str, float]] = []
    for i, start in enumerate(range(0, len(audio), chunk_len)):
        chunk = audio[start : start + chunk_len]
        if len(chunk) < sr // 4:
            continue
        out = tmpdir / f"{Path(path).stem}_{i:05d}.wav"
        sf.write(out, chunk, sr)
        chunks.append((str(out), len(chunk) / sr))
    return chunks


def load_engine(args: argparse.Namespace) -> tuple[Any, float]:
    t0 = time.perf_counter()
    if args.engine in {"faster-whisper", "ctranslate2"}:
        bundle = load_faster_whisper(args.model_path, args.precision)
    else:
        bundle = load_transformers(args.model_path, args.precision)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return bundle, time.perf_counter() - t0


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": props.total_memory / (1024 * 1024),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def run(args: argparse.Namespace) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_dataset(args.dataset)
    samples = load_samples(dataset_path, args.max_samples)
    if not samples:
        raise RuntimeError(f"No benchmark samples loaded from {dataset_path}")

    output_json = Path(args.output_json) if args.output_json else RESULTS_DIR / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.engine}_{Path(str(args.model_path)).name}_{dataset_path.stem}_{args.mode}.json"
    )
    gpu_csv = Path(args.gpu_metrics_csv) if args.gpu_metrics_csv else DEFAULT_GPU_CSV
    sampler = TelemetrySampler(gpu_csv, interval=args.sample_interval)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bundle, startup = load_engine(args)
    sampler.start()
    t0 = time.perf_counter()
    latencies: list[float] = []
    refs: list[str] = []
    preds: list[str] = []
    total_audio = 0.0
    errors: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="whisper_bench_") as td:
        tmpdir = Path(td)
        for sample in samples:
            paths = sample.segments or [sample.audio_path]
            if args.mode == "streaming":
                chunk_items: list[tuple[str, float]] = []
                for p in paths:
                    chunk_items.extend(split_to_temp_wavs(p, args.stream_chunk_seconds, tmpdir))
            else:
                chunk_items = [(p, read_duration(p) or sample.duration / max(len(paths), 1)) for p in paths]

            sample_preds: list[str] = []
            for p, dur in chunk_items:
                try:
                    start = time.perf_counter()
                    pred = transcribe_path(args.engine, bundle, p, args)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - start
                    latencies.append(latency)
                    sample_preds.append(pred)
                    total_audio += float(dur)
                except Exception as exc:
                    errors.append({"audio_path": p, "error": repr(exc)})
            if sample.text:
                refs.append(sample.text)
                preds.append(" ".join(sample_preds).strip())

    processing = time.perf_counter() - t0
    sampler.stop()
    total_wall = startup + processing
    rtf = processing / total_audio if total_audio > 0 else None
    speed = (1.0 / rtf) if rtf and rtf > 0 else None
    throughput = total_audio / processing if processing > 0 else None
    quality = {
        "wer": float(wer(refs, preds)) if refs and preds else None,
        "cer": float(cer(refs, preds)) if refs and preds else None,
        "examples": [
            {"reference": r, "prediction": p}
            for r, p in list(zip(refs, preds))[:20]
        ],
    }
    peak_torch = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None

    result: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "engine": args.engine,
        "model_path": args.model_path,
        "dataset": str(dataset_path),
        "mode": args.mode,
        "precision": args.precision,
        "beam_size": args.beam_size,
        "batch_size": args.batch_size,
        "chunk_size_seconds": args.chunk_size,
        "stream_chunk_seconds": args.stream_chunk_seconds if args.mode == "streaming" else None,
        "vad": args.vad,
        "sample_count": len(samples),
        "error_count": len(errors),
        "errors": errors[:20],
        "timing": {
            "total_audio_seconds": total_audio,
            "total_wall_seconds": total_wall,
            "startup_overhead_seconds": startup,
            "processing_seconds": processing,
            "avg_latency_seconds": statistics.mean(latencies) if latencies else None,
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "p99_latency_seconds": percentile(latencies, 0.99),
        },
        "performance": {
            "throughput_audio_seconds_per_sec": throughput,
            "throughput_audio_hours_per_hour": throughput,
            "rtf": rtf,
            "speed_multiplier": speed,
        },
        "quality": quality,
        "telemetry": sampler.summary(),
        "torch_peak_allocated_vram_mb": peak_torch,
        "hardware": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_gib": psutil.virtual_memory().total / (1024**3),
            **gpu_info(),
        },
        "gpu_metrics_csv": str(gpu_csv),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"result_json": str(output_json), "rtf": rtf, "speed_multiplier": speed, "wer": quality["wer"], "cer": quality["cer"]}, indent=2))
    if errors:
        print(f"WARNING: {len(errors)} samples/chunks failed; first errors are in result JSON.", file=sys.stderr)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["transformers", "faster-whisper", "ctranslate2"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset", default="smoke")
    p.add_argument("--precision", choices=["fp16", "float16", "float32", "int8", "int8_float16"], default="fp16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--mode", choices=["offline", "streaming"], default="offline")
    p.add_argument("--chunk-size", type=float, default=30.0)
    p.add_argument("--stream-chunk-seconds", type=float, default=5.0)
    p.add_argument("--vad", choices=["on", "off"], default="off")
    p.add_argument("--language", choices=["uz"], default="uz")
    p.add_argument("--task", choices=["transcribe"], default="transcribe")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--output-json", default=None)
    p.add_argument("--gpu-metrics-csv", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
