#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
REPORT_ROOT = PROJECT_ROOT / "reports/lr_search/blockwise"
OUTPUT_ROOT = PROJECT_ROOT / "outputs_lr_search/blockwise"
COARSE_CONFIG = PROJECT_ROOT / "configs/lr_search/blockwise/base_blockwise_coarse.yaml"
MAIN_CONFIG = PROJECT_ROOT / "configs/lr_search/blockwise/base_blockwise_main.yaml"

DECODER_LR = 8e-6
WER_TIE = 0.003
CER_TIE = 0.001


@dataclass(frozen=True)
class Candidate:
    phase: str
    name: str
    a: float
    b: float
    c: float
    d: float


def log(message: str) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
    print(line, flush=True)
    with (REPORT_ROOT / "blockwise_search.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def lr_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.1e}".replace(".0", "").replace("-", "m").replace("+", "").replace(".", "p")


def output_dir(experiment_id: str) -> Path:
    return OUTPUT_ROOT / experiment_id


def metrics_path(experiment_id: str) -> Path:
    return output_dir(experiment_id) / "metrics.json"


def load_metrics(experiment_id: str) -> dict[str, Any] | None:
    path = metrics_path(experiment_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def latest_checkpoint(path: Path) -> Path | None:
    checkpoints = [item for item in path.glob("checkpoint-*") if item.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: int(item.name.rsplit("-", 1)[-1]))


def best_validation(metrics: dict[str, Any]) -> dict[str, Any] | None:
    best = metrics.get("best_validation_metrics")
    if best and best.get("eval_wer") is not None:
        return best
    evaluations = [row for row in metrics.get("validation_curve", []) if row.get("eval_wer") is not None]
    if not evaluations:
        return None
    return min(evaluations, key=lambda row: float(row["eval_wer"]))


def value(metrics: dict[str, Any], key: str, default: float = math.inf) -> float:
    best = best_validation(metrics) or {}
    raw = best.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def stability_assessment(metrics: dict[str, Any]) -> tuple[bool, str]:
    if metrics.get("status") != "completed" or not metrics.get("stable", False):
        return False, "runner reported failure or instability"
    best = best_validation(metrics)
    if not best:
        return False, "missing validation metrics"
    wer = value(metrics, "eval_wer")
    cer = value(metrics, "eval_cer")
    hallucination = value(metrics, "eval_hallucination_rate", 0.0)
    confusion = value(metrics, "eval_language_confusion_rate", 0.0)
    if not math.isfinite(wer) or not math.isfinite(cer):
        return False, "non-finite WER/CER"
    if wer > 2.0:
        return False, f"extreme WER {wer:.4f}"
    if hallucination > 0.05:
        return False, f"hallucination rate {hallucination:.4f}"
    if confusion > 0.05:
        return False, f"language-confusion rate {confusion:.4f}"
    return True, "stable"


def ranking_key(metrics: dict[str, Any]) -> tuple[bool, float, float, float, float]:
    stable, _ = stability_assessment(metrics)
    return (
        not stable,
        value(metrics, "eval_wer"),
        value(metrics, "eval_cer"),
        value(metrics, "eval_hallucination_rate", 0.0),
        value(metrics, "eval_language_confusion_rate", 0.0),
    )


def materially_better(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    candidate_wer = value(candidate, "eval_wer")
    control_wer = value(control, "eval_wer")
    candidate_cer = value(candidate, "eval_cer")
    control_cer = value(control, "eval_cer")
    return candidate_wer < control_wer - WER_TIE or (
        abs(candidate_wer - control_wer) <= WER_TIE and candidate_cer < control_cer - CER_TIE
    )


def overrides(candidate: Candidate, max_steps: int | None = None) -> dict[str, Any]:
    data = {
        "experiment_name": candidate.name,
        "encoder_block_a_lr": candidate.a,
        "encoder_block_b_lr": candidate.b,
        "encoder_block_c_lr": candidate.c,
        "encoder_block_d_lr": candidate.d,
        "decoder_learning_rate": DECODER_LR,
        "learning_rate": DECODER_LR,
    }
    if max_steps is not None:
        data["max_steps"] = max_steps
    return data


def execute(candidate: Candidate, stage: str, config: Path, max_steps: int | None = None) -> dict[str, Any]:
    experiment_id = f"{candidate.phase}_{stage}_{candidate.name}"
    cached = load_metrics(experiment_id)
    if cached:
        log(f"Using cached metrics for {experiment_id}")
        return cached

    path = output_dir(experiment_id)
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts/lr_search/run_experiment.py"),
        "--config",
        str(config),
        "--experiment-id",
        f"blockwise/{experiment_id}",
    ]
    if path.exists() and any(path.iterdir()):
        checkpoint = latest_checkpoint(path)
        if checkpoint:
            command.extend(["--resume", "auto"])
            log(f"Resuming {experiment_id} from {checkpoint}")
        else:
            shutil.rmtree(path)
            log(f"Removed incomplete blockwise output without checkpoint: {path}")

    for key, val in overrides(candidate, max_steps=max_steps).items():
        command.extend(["--set", f"{key}={val}"])

    log(f"Launching {experiment_id}: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    metrics = load_metrics(experiment_id)
    if not metrics:
        raise RuntimeError(f"{experiment_id} failed with return code {result.returncode}")
    if result.returncode != 0:
        train_log = output_dir(experiment_id) / "train.log"
        log_text = train_log.read_text(encoding="utf-8", errors="replace").lower() if train_log.exists() else ""
        if "cuda oom" in log_text or "out of memory" in log_text:
            raise RuntimeError(f"{experiment_id} hit CUDA OOM; stopping Phase 4")
        log(f"{experiment_id} returned {result.returncode}; keeping metrics for rejection/ranking")
    write_reports()
    return metrics


def rows_from_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUTPUT_ROOT.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = best_validation(payload) or {}
        schedule = payload.get("blockwise_schedule") or {}
        stable, note = stability_assessment(payload)
        rows.append(
            {
                "experiment": payload.get("experiment_id", path.parent.name),
                "block_a_lr": payload.get("encoder_block_a_lr", schedule.get("encoder_0_7")),
                "block_b_lr": payload.get("encoder_block_b_lr", schedule.get("encoder_8_15")),
                "block_c_lr": payload.get("encoder_block_c_lr", schedule.get("encoder_16_23")),
                "block_d_lr": payload.get("encoder_block_d_lr", schedule.get("encoder_24_31")),
                "decoder_lr": payload.get("decoder_learning_rate", schedule.get("decoder")),
                "wer": best.get("eval_wer"),
                "cer": best.get("eval_cer"),
                "hallucination": best.get("eval_hallucination_rate"),
                "language_confusion": best.get("eval_language_confusion_rate"),
                "runtime_hours": float(payload.get("runtime_seconds", 0.0)) / 3600.0,
                "stable": stable,
                "status": payload.get("status"),
                "notes": note,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["wer"] is None,
            float(row["wer"]) if row["wer"] is not None else math.inf,
            float(row["cer"]) if row["cer"] is not None else math.inf,
            not row["stable"],
        ),
    )


def write_reports() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = rows_from_metrics()
    fields = [
        "rank",
        "experiment",
        "block_a_lr",
        "block_b_lr",
        "block_c_lr",
        "block_d_lr",
        "decoder_lr",
        "wer",
        "cer",
        "hallucination",
        "language_confusion",
        "runtime_hours",
        "stable",
        "status",
        "notes",
    ]
    with (REPORT_ROOT / "phase4_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            writer.writerow({"rank": idx, **row})

    lines = [
        "# Phase 4 Blockwise LR Comparison",
        "",
        "Ranking order: validation WER, validation CER, then stability. Test metrics are not used.",
        "",
        "| Rank | Experiment | A 0-7 | B 8-15 | C 16-23 | D 24-31 | Decoder | WER | CER | Halluc. | Lang conf. | Runtime h | Stable | Notes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for idx, row in enumerate(rows, 1):
        fmt = lambda x: "" if x is None else f"{float(x):.6f}" if isinstance(x, (int, float)) else str(x)
        lines.append(
            f"| {idx} | {row['experiment']} | {row['block_a_lr']} | {row['block_b_lr']} | "
            f"{row['block_c_lr']} | {row['block_d_lr']} | {row['decoder_lr']} | "
            f"{fmt(row['wer'])} | {fmt(row['cer'])} | {fmt(row['hallucination'])} | "
            f"{fmt(row['language_confusion'])} | {float(row['runtime_hours']):.2f} | "
            f"{'yes' if row['stable'] else 'no'} | {row['notes']} |"
        )
    (REPORT_ROOT / "phase4_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_plots(rows)


def make_plots(rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row["wer"] is not None and row["cer"] is not None]
    if not completed:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib unavailable; skipping Phase 4 plots")
        return
    plot_dir = REPORT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = [Path(row["experiment"]).name for row in completed]
    for metric in ("wer", "cer"):
        plt.figure(figsize=(max(9, len(names) * 0.9), 5))
        plt.bar(names, [float(row[metric]) for row in completed])
        plt.ylabel(metric.upper())
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"blockwise_{metric}_comparison.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    for block, field in (
        ("A 0-7", "block_a_lr"),
        ("B 8-15", "block_b_lr"),
        ("C 16-23", "block_c_lr"),
        ("D 24-31", "block_d_lr"),
    ):
        xs = [max(float(row[field] or 0.0), 1e-8) for row in completed]
        ys = [float(row["wer"]) for row in completed]
        plt.scatter(xs, ys, label=block)
    plt.xscale("log")
    plt.xlabel("Block learning rate")
    plt.ylabel("Validation WER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "block_lr_vs_wer.png", dpi=160)
    plt.close()

    for path in sorted(OUTPUT_ROOT.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        train = payload.get("train_loss_curve", [])
        val = payload.get("validation_curve", [])
        if not train and not val:
            continue
        plt.figure(figsize=(8, 4.5))
        if train:
            plt.plot([row["step"] for row in train], [row["loss"] for row in train], label="train loss")
        val_points = [row for row in val if row.get("eval_loss") is not None]
        if val_points:
            plt.plot([row["step"] for row in val_points], [row["eval_loss"] for row in val_points], label="eval loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"loss_{path.parent.name}.png", dpi=160)
        plt.close()


def select_best(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [item for item in metrics if stability_assessment(item)[0]]
    if not stable:
        raise RuntimeError("No stable blockwise candidates available for selection")
    return min(stable, key=ranking_key)


def run_candidates(candidates: list[Candidate], phase_name: str) -> list[dict[str, Any]]:
    survivors: list[Candidate] = []
    for candidate in candidates:
        screen = execute(candidate, "screen", COARSE_CONFIG, max_steps=300)
        stable, note = stability_assessment(screen)
        log(f"{candidate.name} screen: {note}")
        if stable:
            survivors.append(candidate)
    if not survivors:
        raise RuntimeError(f"{phase_name}: all candidates failed divergence screening")
    full_metrics = []
    for candidate in survivors:
        full_metrics.append(execute(candidate, "full", MAIN_CONFIG))
    write_reports()
    return full_metrics


def find_metrics_by_id(experiment_id: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / "outputs_lr_search" / experiment_id / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def select_best_existing_block_d() -> dict[str, Any]:
    candidates = []
    for experiment_id in (
        "phase2_upper_encoder_5em07",
        "phase2_upper_encoder_1em06",
        "phase2_upper_encoder_2em06",
        "phase2_upper_encoder_5em06",
        "phase2_upper_encoder_8em06",
    ):
        metrics = find_metrics_by_id(experiment_id)
        if metrics:
            candidates.append(metrics)
    if not candidates:
        raise RuntimeError("No completed upper-encoder metrics found for Block D selection")
    best = select_best(candidates)
    encoder_lr = float(best.get("encoder_learning_rate") or 0.0)
    best["encoder_block_a_lr"] = 0.0
    best["encoder_block_b_lr"] = 0.0
    best["encoder_block_c_lr"] = 0.0
    best["encoder_block_d_lr"] = encoder_lr
    best["blockwise_schedule"] = {
        "encoder_0_7": 0.0,
        "encoder_8_15": 0.0,
        "encoder_16_23": 0.0,
        "encoder_24_31": encoder_lr,
        "decoder": best.get("decoder_learning_rate"),
    }
    return best


def final_comparison_rows(selected: dict[str, Any]) -> list[str]:
    rows = []
    references = [
        ("Blockwise winner", selected, "30h proxy validation"),
        ("Decoder-only search", find_metrics_by_id("phase2_decoder_8em06"), "30h proxy validation"),
        ("Single encoder LR", find_metrics_by_id("phase2_upper_encoder_5em06"), "30h proxy validation"),
        ("Half encoder single LR", find_metrics_by_id("phase3_freeze_boundary_15"), "30h proxy validation"),
    ]
    baseline_path = PROJECT_ROOT / "models/partial_ft_usc_baseline/metrics/test_metrics.json"
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        references.append(("Partial FT baseline", {"test": payload}, "USC test"))
    full_path = PROJECT_ROOT / "outputs_full_ft/test_metrics.json"
    if full_path.exists():
        payload = json.loads(full_path.read_text(encoding="utf-8"))
        references.append(("Full FT USC", {"test": payload}, "USC test"))

    for name, payload, dataset in references:
        if not payload:
            rows.append(f"| {name} | {dataset} | unavailable | unavailable | unavailable |")
            continue
        if "test" in payload:
            metrics = payload["test"]
            wer = metrics.get("test_wer") or metrics.get("wer")
            cer = metrics.get("test_cer") or metrics.get("cer")
        else:
            best = best_validation(payload) or {}
            wer = best.get("eval_wer")
            cer = best.get("eval_cer")
        rows.append(f"| {name} | {dataset} | {wer} | {cer} | see source metrics |")
    return rows


def write_final(selected: dict[str, Any]) -> None:
    schedule = selected.get("blockwise_schedule") or {}
    best = best_validation(selected) or {}
    lines = [
        "# Final Blockwise LR Recommendation",
        "",
        "## Recommended Schedule",
        "",
        f"- decoder = `{schedule.get('decoder', selected.get('decoder_learning_rate'))}`",
        f"- enc24-31 = `{schedule.get('encoder_24_31', selected.get('encoder_block_d_lr'))}`",
        f"- enc16-23 = `{schedule.get('encoder_16_23', selected.get('encoder_block_c_lr'))}`",
        f"- enc8-15 = `{schedule.get('encoder_8_15', selected.get('encoder_block_b_lr'))}`",
        f"- enc0-7 = `{schedule.get('encoder_0_7', selected.get('encoder_block_a_lr'))}`",
        "",
        "## Winning Evidence",
        "",
        f"- Experiment: `{selected.get('experiment_id')}`",
        f"- Validation WER: `{best.get('eval_wer')}`",
        f"- Validation CER: `{best.get('eval_cer')}`",
        f"- Hallucination rate: `{best.get('eval_hallucination_rate')}`",
        f"- Language-confusion rate: `{best.get('eval_language_confusion_rate')}`",
        "",
        "## Comparison",
        "",
        "| Regime | Dataset/metric source | WER | CER | Notes |",
        "|---|---|---:|---:|---|",
        *final_comparison_rows(selected),
        "",
        "Proxy-validation results are used for hyperparameter selection. USC test metrics are",
        "historical references and are not directly comparable to the 30h proxy validation set.",
        "",
    ]
    (REPORT_ROOT / "FINAL_BLOCKWISE_RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PYTHON), str(PROJECT_ROOT / "scripts/lr_search/audit_data_leakage.py")], cwd=PROJECT_ROOT, check=True)

    best_d = select_best_existing_block_d()
    best_d_lr = float(best_d["encoder_block_d_lr"])
    log(f"Phase 4A skipped; selected existing upper-encoder D LR={best_d_lr}")

    c_candidates = [
        Candidate("phase4b", f"block_c_{lr_label(lr)}_d_{lr_label(best_d_lr)}", 0.0, 0.0, lr, best_d_lr)
        for lr in (5e-7, 1e-6, 2e-6, 5e-6, 8e-6)
        if lr <= best_d_lr
    ]
    c_metrics = [best_d] + run_candidates(c_candidates, "Phase 4B") if c_candidates else [best_d]
    best_c = select_best(c_metrics)
    best_c_lr = float(best_c.get("encoder_block_c_lr") or 0.0)
    log(f"Phase 4B selected C LR={best_c_lr}")

    b_candidates = [
        Candidate("phase4c", f"block_b_{lr_label(lr)}_c_{lr_label(best_c_lr)}_d_{lr_label(best_d_lr)}", 0.0, lr, best_c_lr, best_d_lr)
        for lr in (1e-7, 5e-7, 1e-6, 2e-6)
        if best_c_lr > 0.0 and lr <= best_c_lr
    ]
    b_metrics = [best_c] + run_candidates(b_candidates, "Phase 4C") if b_candidates else [best_c]
    best_b = select_best(b_metrics)
    best_b_lr = float(best_b.get("encoder_block_b_lr") or 0.0)
    log(f"Phase 4C selected B LR={best_b_lr}")

    selected = best_b
    if best_b_lr > 0.0 and materially_better(best_b, best_c):
        a_candidates = [
            Candidate("phase4d", f"block_a_{lr_label(lr)}_b_{lr_label(best_b_lr)}_c_{lr_label(best_c_lr)}_d_{lr_label(best_d_lr)}", lr, best_b_lr, best_c_lr, best_d_lr)
            for lr in (1e-7, 5e-7)
            if lr <= best_b_lr
        ]
        a_metrics = [best_b] + run_candidates(a_candidates, "Phase 4D") if a_candidates else [best_b]
        selected = select_best(a_metrics)
        log(f"Phase 4D selected A LR={selected.get('encoder_block_a_lr')}")
    else:
        log("Phase 4D skipped: Block B did not provide material evidence for lower-block adaptation")

    write_reports()
    write_final(selected)
    log("Phase 4 blockwise search completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        raise
