#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = config.pop("base_config", None)
    if not base:
        return config
    base_path = Path(base).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = load_config(base_path.resolve())
    merged.update(config)
    return merged


def parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"Override must use KEY=VALUE syntax: {value!r}")
    key, raw = value.split("=", 1)
    return key, yaml.safe_load(raw)


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model_name",
        "data_dir",
        "tuning_mode",
        "decoder_learning_rate",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "eval_steps",
        "save_steps",
    }
    missing = sorted(key for key in required if config.get(key) is None)
    if missing:
        raise ValueError(f"Resolved config is missing required values: {missing}")
    if config["tuning_mode"] not in {
        "decoder_only",
        "encoder_24_31_plus_decoder",
        "encoder_16_31_plus_decoder",
    }:
        raise ValueError(f"Unsupported LR-search tuning_mode: {config['tuning_mode']!r}")
    if float(config["decoder_learning_rate"]) <= 0:
        raise ValueError("decoder_learning_rate must be positive")
    if config["tuning_mode"] != "decoder_only" and float(config.get("encoder_learning_rate", 0)) <= 0:
        raise ValueError("encoder_learning_rate must be positive when encoder layers are trainable")
    if int(config["save_steps"]) % int(config["eval_steps"]) != 0:
        raise ValueError("save_steps must be a multiple of eval_steps when load_best_model_at_end is enabled")
    data_dir = Path(config["data_dir"]).expanduser()
    for split in ("train", "val", "test"):
        if not (data_dir / f"{split}.csv").is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {data_dir / f'{split}.csv'}")
    if not bool(config.get("bf16")) or bool(config.get("fp16")):
        raise ValueError("LR-search configs must use bf16=true and fp16=false")
    if config.get("language") != "uz" or config.get("task") != "transcribe":
        raise ValueError("LR-search configs must force language='uz' and task='transcribe'")
    if bool(config.get("evaluate_test_after_training", True)) or bool(config.get("load_test_split", True)):
        raise ValueError("LR-search runs must neither load nor evaluate the test split")


def sample_gpu(stop: threading.Event, output: Path, interval: float = 5.0) -> None:
    fields = [
        "timestamp",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
        "temperature.gpu",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "gpu_util_percent", "vram_used_mib", "vram_total_mib", "power_w", "temp_c"])
        while not stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={','.join(fields)}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                for line in result.stdout.strip().splitlines():
                    writer.writerow([item.strip() for item in line.split(",")])
                handle.flush()
            except Exception as exc:
                writer.writerow([datetime.now(timezone.utc).isoformat(), "", "", "", "", f"error:{exc!r}"])
                handle.flush()
            stop.wait(interval)


def gpu_summary(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    def values(key: str) -> list[float]:
        result = []
        for row in rows:
            try:
                result.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return result

    util = values("gpu_util_percent")
    vram = values("vram_used_mib")
    return {
        "samples": len(rows),
        "average_gpu_util_percent": sum(util) / len(util) if util else None,
        "peak_gpu_util_percent": max(util) if util else None,
        "average_vram_mib": sum(vram) / len(vram) if vram else None,
        "peak_vram_mib": max(vram) if vram else None,
    }


def save_experiment_plots(run_metrics: dict[str, Any], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    history = run_metrics.get("log_history", [])
    series = {
        "train_loss": [(item["step"], item["loss"]) for item in history if "loss" in item and "step" in item],
        "val_loss": [(item["step"], item["eval_loss"]) for item in history if "eval_loss" in item and "step" in item],
    }
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, points in series.items():
        if not points:
            continue
        plt.figure(figsize=(8, 4.5))
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=2)
        plt.xlabel("Step")
        plt.ylabel(name.replace("_", " ").title())
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png", dpi=160)
        plt.close()


def collect_metrics(
    output_dir: Path,
    config: dict[str, Any],
    experiment_id: str,
    return_code: int,
    started_at: str,
    elapsed: float,
) -> dict[str, Any]:
    run_metrics_path = output_dir / "run_metrics.json"
    run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8")) if run_metrics_path.exists() else {}
    if run_metrics.get("test_metrics"):
        raise RuntimeError("Benchmark integrity violation: LR-search run produced test metrics")
    log_history = run_metrics.get("log_history", [])
    losses = [{"step": item.get("step"), "loss": item["loss"]} for item in log_history if "loss" in item]
    evaluations = [
        {
            key: item.get(key)
            for key in (
                "step",
                "eval_loss",
                "eval_wer",
                "eval_cer",
                "eval_hallucination_rate",
                "eval_language_confusion_rate",
            )
        }
        for item in log_history
        if "eval_loss" in item
    ]
    best_validation = (
        min(
            (row for row in evaluations if row.get("eval_wer") is not None),
            key=lambda row: float(row["eval_wer"]),
            default=None,
        )
    )
    text = (output_dir / "train.log").read_text(encoding="utf-8", errors="replace") if (output_dir / "train.log").exists() else ""
    lower_log = text.lower()
    stability_failures = [
        token
        for token in ("safety_stop", "cuda oom", "non-finite loss")
        if token in lower_log
    ]
    safety_warning_count = lower_log.count("safety_warning")
    stable = return_code == 0 and not stability_failures
    payload = {
        "experiment_id": experiment_id,
        "experiment_name": config.get("experiment_name", experiment_id),
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "started_at_utc": started_at,
        "runtime_seconds": elapsed,
        "tuning_mode": config.get("tuning_mode"),
        "encoder_learning_rate": config.get("encoder_learning_rate"),
        "decoder_learning_rate": config.get("decoder_learning_rate"),
        "dataset": config.get("data_dir"),
        "stable": stable,
        "stability_failures": stability_failures,
        "safety_warning_count": safety_warning_count,
        "train_loss_curve": losses,
        "validation_curve": evaluations,
        "best_validation_metrics": best_validation,
        "test_metrics": run_metrics.get("test_metrics", {}),
        "best_checkpoint_step": (
            int(Path(run_metrics["best_checkpoint"]).name.split("-")[-1])
            if run_metrics.get("best_checkpoint")
            else None
        ),
        "final_checkpoint_step": run_metrics.get("final_step"),
        "best_metric": run_metrics.get("best_metric"),
        "gpu": gpu_summary(output_dir / "gpu_metrics.csv"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_experiment_plots(run_metrics, output_dir / "plots")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one reproducible Whisper LR-search experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", help="Stable output directory name; required to resume an existing run.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume from a checkpoint path or latest checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate config without launching training.")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    for value in args.overrides:
        key, parsed = parse_override(value)
        config[key] = parsed
    validate_config(config)
    base_name = str(config.get("experiment_name") or config_path.stem)
    experiment_id = args.experiment_id or base_name
    output_dir = PROJECT_ROOT / "outputs_lr_search" / experiment_id
    if args.dry_run:
        config["output_dir"] = str(output_dir)
        config["logging_dir"] = str(output_dir / "logs")
        config["status_report_dir"] = str(output_dir / "status_reports")
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None and not args.dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"{base_name}__{stamp}"
        output_dir = PROJECT_ROOT / "outputs_lr_search" / experiment_id

    config["output_dir"] = str(output_dir)
    config["logging_dir"] = str(output_dir / "logs")
    config["status_report_dir"] = str(output_dir / "status_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata = {
        "experiment_id": experiment_id,
        "source_config": str(config_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip() or None,
        "command": sys.argv,
    }
    (output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    command = [str(PROJECT_ROOT / ".venv/bin/python"), str(PROJECT_ROOT / "src/train.py"), "--config", str(resolved_path)]
    if args.resume:
        command.extend(["--resume", args.resume])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["TOKENIZERS_PARALLELISM"] = "false"

    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, output_dir / "gpu_metrics.csv"), daemon=True)
    monitor.start()
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with (output_dir / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def forward_signal(signum, _frame):
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    elapsed = time.monotonic() - start
    stop.set()
    monitor.join(timeout=10)
    metrics = collect_metrics(output_dir, config, experiment_id, return_code, started_at, elapsed)
    print(json.dumps(metrics, indent=2))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
