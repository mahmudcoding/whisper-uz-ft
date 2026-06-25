#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import yaml
from faster_whisper import WhisperModel

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent USC-only teacher agreement for SILVER.")
    parser.add_argument("--config", default=str(ROOT / "configs/silver_datasets.yaml"))
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    work_dir = Path(cfg["work_dir"]).expanduser()
    report_dir = Path(cfg["report_dir"]).expanduser()
    teacher_path = Path(cfg["teacher_model"]).expanduser()
    input_path = work_dir / "silver_teacher_candidates.csv"
    output_path = work_dir / "silver_teacher_scored.csv"
    rejected_path = report_dir / "teacher_rejected.csv"
    if not teacher_path.joinpath("model.bin").exists():
        raise FileNotFoundError(f"Missing converted teacher model: {teacher_path}")
    if not args.no_wait:
        wait_for_gpu()

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        input_fields = list(csv.DictReader(handle).fieldnames or [])
    extra_fields = [
        "teacher_text", "teacher_language", "teacher_language_probability",
        "teacher_similarity", "teacher_wer", "teacher_cer", "teacher_quality_score",
        "teacher_decision", "teacher_rejection_reasons", "teacher_error",
    ]
    fields = list(dict.fromkeys(input_fields + extra_fields))
    completed = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            completed = max(0, sum(1 for _ in handle) - 1)

    print(f"LOAD_TEACHER path={teacher_path} completed_rows={completed}", flush=True)
    model = WhisperModel(
        str(teacher_path),
        device=cfg["teacher_device"],
        compute_type=cfg["teacher_compute_type"],
    )
    thresholds = cfg["thresholds"]
    mode = "a" if completed else "w"
    rejected_mode = "a" if rejected_path.exists() else "w"
    with input_path.open("r", encoding="utf-8", newline="") as input_handle, output_path.open(
        mode, encoding="utf-8", newline=""
    ) as output_handle, rejected_path.open(rejected_mode, encoding="utf-8", newline="") as rejected_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        rejected_writer = csv.DictWriter(rejected_handle, fieldnames=fields)
        if not completed:
            writer.writeheader()
        if rejected_mode == "w":
            rejected_writer.writeheader()
        kept = 0
        rejected = 0
        for idx, row in enumerate(reader):
            if idx < completed:
                continue
            reasons: list[str] = []
            error = ""
            try:
                segments, info = model.transcribe(
                    row["audio_path"],
                    language=None,
                    task="transcribe",
                    beam_size=int(cfg["teacher_beam_size"]),
                    best_of=1,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=False,
                    without_timestamps=True,
                )
                teacher_text = normalize_uzbek_text(" ".join(segment.text.strip() for segment in segments))
                language = str(info.language or "")
                language_probability = float(info.language_probability or 0.0)
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
            if language != "uz":
                score -= 35
                reasons.append(f"teacher_language_{language or 'unknown'}")
            if language_probability < float(thresholds["min_language_probability"]):
                score -= 20
                reasons.append("low_uzbek_language_confidence")
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
            row.update(
                {
                    "teacher_text": teacher_text,
                    "teacher_language": language,
                    "teacher_language_probability": language_probability,
                    "teacher_similarity": similarity,
                    "teacher_wer": wer,
                    "teacher_cer": cer,
                    "teacher_quality_score": score,
                    "teacher_decision": decision,
                    "teacher_rejection_reasons": "|".join(sorted(set(reasons))),
                    "teacher_error": error,
                }
            )
            writer.writerow(row)
            if decision == "keep":
                kept += 1
            else:
                rejected += 1
                rejected_writer.writerow(row)
            if (idx + 1) % 100 == 0:
                output_handle.flush()
                rejected_handle.flush()
                print(
                    f"TEACHER processed={idx + 1} kept_this_run={kept} "
                    f"rejected_this_run={rejected}",
                    flush=True,
                )
    summary = {
        "teacher_model": str(teacher_path),
        "teacher_independence": "USC-only partial fine-tune; no SILVER training exposure",
        "input": str(input_path),
        "output": str(output_path),
        "thresholds": thresholds,
    }
    (report_dir / "teacher_scoring_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
