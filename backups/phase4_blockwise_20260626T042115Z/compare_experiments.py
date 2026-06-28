#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def metric(payload: dict[str, Any], name: str) -> float | None:
    if payload.get("test_metrics"):
        raise ValueError(
            f"{payload.get('experiment_id')}: test metrics are forbidden in LR-search comparison"
        )
    evaluations = [row for row in payload.get("validation_curve", []) if row.get(f"eval_{name}") is not None]
    if not evaluations:
        return None
    if name == "wer":
        return min(float(row["eval_wer"]) for row in evaluations)
    best_wer_row = min(evaluations, key=lambda row: float(row.get("eval_wer", float("inf"))))
    return float(best_wer_row[f"eval_{name}"])


def load_experiments(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    metric_paths = [path / "metrics.json" if path.is_dir() else path for path in paths]
    if not metric_paths:
        metric_paths = sorted(root.glob("*/metrics.json"))
    rows = []
    for path in metric_paths:
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": payload.get("experiment_name"),
                "experiment_id": payload.get("experiment_id"),
                "trainable_layers": payload.get("tuning_mode"),
                "decoder_lr": payload.get("decoder_learning_rate"),
                "encoder_lr": payload.get("encoder_learning_rate"),
                "best_wer": metric(payload, "wer"),
                "best_cer": metric(payload, "cer"),
                "runtime_hours": float(payload.get("runtime_seconds", 0)) / 3600.0,
                "peak_vram_mib": payload.get("gpu", {}).get("peak_vram_mib"),
                "stable": bool(payload.get("stable")),
                "status": payload.get("status"),
                "notes": "" if payload.get("stable") else "failed or instability marker detected",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["best_wer"] is None,
            row["best_wer"] if row["best_wer"] is not None else float("inf"),
            row["best_cer"] if row["best_cer"] is not None else float("inf"),
            not row["stable"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "experiment_name", "experiment_id", "trainable_layers", "decoder_lr", "encoder_lr",
        "best_wer", "best_cer", "runtime_hours", "peak_vram_mib", "stable", "status", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LR Search Experiment Comparison",
        "",
        "Ranking order: validation WER, validation CER, then stability.",
        "Test metrics are forbidden during LR search.",
        "",
        "| Rank | Experiment | Trainable layers | Decoder LR | Encoder LR | WER | CER | Runtime h | Stable | Notes |",
        "|---:|---|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for rank, row in enumerate(rows, 1):
        format_metric = lambda value: "" if value is None else f"{value:.6f}"
        lines.append(
            f"| {rank} | {row['experiment_id']} | {row['trainable_layers']} | {row['decoder_lr']} | "
            f"{row['encoder_lr']} | {format_metric(row['best_wer'])} | {format_metric(row['best_cer'])} | "
            f"{row['runtime_hours']:.2f} | {'yes' if row['stable'] else 'no'} | {row['notes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    completed = [row for row in rows if row["best_wer"] is not None and row["best_cer"] is not None]
    if not completed:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; skipping comparison plots")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [row["experiment_id"] for row in completed]
    for metric_name in ("best_wer", "best_cer"):
        plt.figure(figsize=(max(8, len(names) * 1.1), 5))
        plt.bar(names, [row[metric_name] for row in completed])
        plt.ylabel(metric_name.replace("best_", "").upper())
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric_name.replace('best_', '')}_comparison.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 5))
    for mode in sorted({row["trainable_layers"] for row in completed}):
        subset = [row for row in completed if row["trainable_layers"] == mode]
        x = [
            float(row["decoder_lr"] if mode == "decoder_only" else row["encoder_lr"])
            for row in subset
        ]
        y = [row["best_wer"] for row in subset]
        plt.scatter(x, y, label=mode)
        for x_value, y_value, row in zip(x, y, subset):
            plt.annotate(row["experiment_id"], (x_value, y_value), fontsize=7)
    plt.xscale("log")
    plt.xlabel("Searched learning rate")
    plt.ylabel("WER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lr_vs_wer.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank LR-search experiments and generate plots.")
    parser.add_argument("experiments", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path("outputs_lr_search"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/lr_search"))
    args = parser.parse_args()
    rows = load_experiments(args.experiments, args.root)
    write_csv(args.report_dir / "experiment_comparison.csv", rows)
    write_markdown(args.report_dir / "experiment_comparison.md", rows)
    make_plots(rows, args.report_dir / "plots")
    print(f"Compared {len(rows)} experiments")


if __name__ == "__main__":
    main()
