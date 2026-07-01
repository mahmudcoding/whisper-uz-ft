#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import subprocess
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any

import yaml
from faster_whisper import BatchedInferencePipeline, WhisperModel

from filtering.similarity import normalized_cer, normalized_similarity, normalized_wer
from text_normalization import normalize_uzbek_text


ROOT = Path(__file__).resolve().parents[2]


def gpu_snapshot() -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    utilization, memory = result.stdout.strip().split(",")[:2]
    return float(utilization.strip()), float(memory.strip())


def wait_for_gpu(max_memory_mib: float = 8000, max_utilization: float = 20) -> None:
    stable = 0
    while stable < 3:
        utilization, memory = gpu_snapshot()
        if memory <= max_memory_mib and utilization <= max_utilization:
            stable += 1
        else:
            stable = 0
        print(
            f"WAIT_GPU utilization={utilization:.0f}% memory={memory:.0f}MiB stable={stable}/3",
            flush=True,
        )
        if stable < 3:
            time.sleep(60)


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["audio_path"] for row in csv.DictReader(handle) if row.get("audio_path")}


def ensure_header(path: Path, fields: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def append_row(path: Path, lock_path: Path, fields: list[str], row: dict[str, Any]) -> None:
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(
                {field: row.get(field, "") for field in fields}
            )
        fcntl.flock(lock, fcntl.LOCK_UN)


def score_row(
    row: dict[str, str],
    pipeline: BatchedInferencePipeline,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    thresholds = cfg["thresholds"]
    forced_language = str(cfg.get("teacher_force_language") or "uz")
    task = str(cfg.get("teacher_task") or "transcribe")
    reasons: list[str] = []
    error = ""
    try:
        segments, info = pipeline.transcribe(
            row["audio_path"],
            language=forced_language,
            task=task,
            beam_size=int(cfg["teacher_beam_size"]),
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=False,
            without_timestamps=True,
            batch_size=int(cfg.get("teacher_batch_size", 16)),
        )
        teacher_text = normalize_uzbek_text(
            " ".join(segment.text.strip() for segment in segments)
        )
        language = forced_language
        language_probability = float(info.language_probability or 1.0)
        similarity = normalized_similarity(row["normalized_text"], teacher_text)
        wer = normalized_wer(row["normalized_text"], teacher_text)
        cer = normalized_cer(row["normalized_text"], teacher_text)
    except Exception as exc:
        teacher_text = ""
        language = ""
        language_probability = 0.0
        similarity = 0.0
        wer = 10.0
        cer = 10.0
        error = repr(exc)
        reasons.append("teacher_error")

    score = float(row["heuristic_quality_score"])
    if similarity < float(thresholds["min_teacher_similarity"]):
        score -= 30
        reasons.append("low_teacher_similarity")
    if wer > float(thresholds["max_teacher_wer"]):
        score -= 20
        reasons.append("high_teacher_wer")
    if cer > float(thresholds["max_teacher_cer"]):
        score -= 20
        reasons.append("high_teacher_cer")
    score = max(0.0, min(100.0, score))
    decision = (
        "keep"
        if not reasons and score >= float(thresholds["min_final_quality_score"])
        else "reject"
    )
    return {
        **row,
        "teacher_text": teacher_text,
        "teacher_language": language,
        "teacher_language_probability": language_probability,
        "teacher_forced_language": forced_language,
        "teacher_task": task,
        "teacher_similarity": similarity,
        "teacher_wer": wer,
        "teacher_cer": cer,
        "teacher_quality_score": score,
        "teacher_decision": decision,
        "teacher_rejection_reasons": "|".join(sorted(set(reasons))),
        "teacher_error": error,
    }


def worker(
    shard_index: int,
    num_shards: int,
    cfg: dict[str, Any],
    fields: list[str],
    completed: set[str],
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    lock_path: Path,
) -> None:
    model = WhisperModel(
        str(Path(cfg["teacher_model"]).expanduser()),
        device=cfg["teacher_device"],
        compute_type=cfg["teacher_compute_type"],
        num_workers=int(cfg.get("teacher_model_num_workers", 1)),
    )
    pipeline = BatchedInferencePipeline(model=model)
    kept = 0
    rejected = 0
    processed = 0
    with input_path.open("r", encoding="utf-8", newline="") as input_handle:
        for idx, row in enumerate(csv.DictReader(input_handle)):
            if idx % num_shards != shard_index:
                continue
            if row.get("audio_path") in completed:
                continue
            result = score_row(row, pipeline, cfg)
            append_row(output_path, lock_path, fields, result)
            if result["teacher_decision"] == "keep":
                kept += 1
            else:
                rejected += 1
                append_row(rejected_path, lock_path, fields, result)
            processed += 1
            if processed % 100 == 0:
                print(
                    f"TEACHER_SHARD shard={shard_index}/{num_shards} "
                    f"processed_this_worker={processed} kept={kept} rejected={rejected}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Kotib teacher agreement for SILVER.")
    parser.add_argument("--config", default=str(ROOT / "configs/silver_datasets.yaml"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    work_dir = Path(cfg["work_dir"]).expanduser()
    report_dir = Path(cfg["report_dir"]).expanduser()
    input_path = work_dir / "silver_teacher_candidates.csv"
    output_path = work_dir / "silver_teacher_scored.csv"
    rejected_path = report_dir / "teacher_rejected.csv"
    teacher_path = Path(cfg["teacher_model"]).expanduser()
    if not teacher_path.joinpath("model.bin").exists():
        raise FileNotFoundError(f"Missing converted teacher model: {teacher_path}")
    if not args.no_wait:
        wait_for_gpu()

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        input_fields = list(csv.DictReader(handle).fieldnames or [])
    extra_fields = [
        "teacher_text", "teacher_language", "teacher_language_probability",
        "teacher_forced_language", "teacher_task",
        "teacher_similarity", "teacher_wer", "teacher_cer", "teacher_quality_score",
        "teacher_decision", "teacher_rejection_reasons", "teacher_error",
    ]
    fields = list(dict.fromkeys(input_fields + extra_fields))
    ensure_header(output_path, fields)
    ensure_header(rejected_path, fields)
    completed = read_completed(output_path)
    total = max(0, sum(1 for _ in input_path.open("r", encoding="utf-8")) - 1)
    workers = int(args.workers or cfg.get("teacher_parallel_workers", 4))
    print(
        f"LOAD_TEACHER_PARALLEL path={teacher_path} completed_rows={len(completed)} "
        f"total_candidates={total} workers={workers} batch_size={cfg.get('teacher_batch_size', 16)}",
        flush=True,
    )
    summary = {
        "teacher_hf_id": cfg["teacher_hf_id"],
        "teacher_revision": cfg["teacher_revision"],
        "teacher_model": str(teacher_path),
        "teacher_policy": (
            "Kotib Uzbek-only Whisper Medium teacher with forced Uzbek transcription; "
            "automatic Whisper language detection is not used as a quality gate"
        ),
        "forced_language": str(cfg.get("teacher_force_language") or "uz"),
        "task": str(cfg.get("teacher_task") or "transcribe"),
        "parallel_workers": workers,
        "batch_size": int(cfg.get("teacher_batch_size", 16)),
        "input": str(input_path),
        "output": str(output_path),
        "thresholds": cfg["thresholds"],
    }
    (report_dir / "teacher_scoring_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lock_path = work_dir / "silver_teacher_scored.lock"
    processes = [
        Process(
            target=worker,
            args=(
                shard,
                workers,
                cfg,
                fields,
                completed,
                input_path,
                output_path,
                rejected_path,
                lock_path,
            ),
        )
        for shard in range(workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        if process.exitcode:
            raise SystemExit(f"Worker failed with exit code {process.exitcode}")
    print("TEACHER_PARALLEL_DONE", flush=True)


if __name__ == "__main__":
    main()
