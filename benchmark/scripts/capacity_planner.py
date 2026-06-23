#!/usr/bin/env python3
"""Capacity and cost planner using measured Whisper benchmark results."""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COSTS = ROOT / "benchmark" / "configs" / "hardware_costs.yaml"
DEFAULT_REPORT = ROOT / "benchmark" / "reports" / "final_capacity_report.md"
DEFAULT_JSON = ROOT / "benchmark" / "reports" / "capacity_plan.json"


def load_results(pattern: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(path).read_text())
            if data.get("performance", {}).get("speed_multiplier"):
                data["_path"] = path
                results.append(data)
        except Exception:
            continue
    if not results:
        raise FileNotFoundError(f"No usable benchmark results found for {pattern}")
    return results


def best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda r: float(r["performance"]["speed_multiplier"]))


def pipeline_multiplier(cfg: dict[str, Any]) -> tuple[float, float, list[str]]:
    latency = 1.0
    cost = 1.0
    enabled = []
    for name, spec in cfg.get("pipeline_stages", {}).items():
        if bool(spec.get("enabled", False)):
            enabled.append(name)
            latency += float(spec.get("latency_multiplier", 0.0))
            cost += float(spec.get("cost_multiplier", 0.0))
    return latency, cost, enabled


def plan_for_hardware(
    measured_speed: float,
    measured_audio_hph: float,
    hw_name: str,
    hw: dict[str, Any],
    scenarios: list[int],
    utilization: float,
    redundancy: float,
    cost_mode: str,
    latency_multiplier: float,
    cost_multiplier: float,
) -> dict[str, Any]:
    rel_perf = float(hw.get("relative_perf_vs_a40", 1.0))
    speed = measured_speed * rel_perf / latency_multiplier
    audio_hph = measured_audio_hph * rel_perf / latency_multiplier
    streams_per_gpu = speed * utilization
    gpu_hour_cost = float(hw.get("pricing", {}).get(cost_mode, hw.get("pricing", {}).get("on_demand", 0.0))) * cost_multiplier
    scenarios_out = {}
    for streams in scenarios:
        required_gpus = math.ceil((streams * redundancy) / max(streams_per_gpu, 1e-9))
        servers = math.ceil(required_gpus / int(hw.get("gpus_per_server", 1)))
        monthly = required_gpus * gpu_hour_cost * 730
        annual = monthly * 12
        scenarios_out[str(streams)] = {
            "target_concurrent_streams": streams,
            "required_gpus": required_gpus,
            "required_servers": servers,
            "monthly_cost_usd": monthly,
            "annual_cost_usd": annual,
        }
    cost_per_audio_hour = gpu_hour_cost / max(audio_hph * utilization, 1e-9)
    return {
        "hardware": hw_name,
        "gpu": hw.get("gpu"),
        "relative_perf_vs_a40": rel_perf,
        "estimated_speed_multiplier": speed,
        "estimated_audio_hours_per_hour_per_gpu": audio_hph,
        "streams_per_gpu_at_target_utilization": streams_per_gpu,
        "gpu_hour_cost_usd": gpu_hour_cost,
        "cost_per_audio_hour_usd": cost_per_audio_hour,
        "scenarios": scenarios_out,
    }


def money(v: float) -> str:
    return f"${v:,.2f}"


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    best = plan["best_benchmark"]
    lines = [
        "# Whisper Inference Capacity Report",
        "",
        f"Generated: {plan['created_at_utc']}",
        "",
        "## Benchmark Summary",
        "",
        f"- Result file: `{best['_path']}`",
        f"- Engine: `{best['engine']}`",
        f"- Model: `{best['model_path']}`",
        f"- Dataset: `{best['dataset']}`",
        f"- Mode: `{best['mode']}`",
        f"- Precision: `{best['precision']}`",
        f"- Beam size: `{best['beam_size']}`",
        f"- Batch size: `{best['batch_size']}`",
        f"- Total audio measured: {best['timing']['total_audio_seconds']:.2f} seconds",
        f"- Processing time: {best['timing']['processing_seconds']:.2f} seconds",
        f"- RTF: {best['performance']['rtf']:.4f}",
        f"- Speed multiplier: {best['performance']['speed_multiplier']:.2f}x real time",
        f"- Throughput on measured GPU: {best['performance']['throughput_audio_hours_per_hour']:.2f} audio-hours/hour",
        f"- WER: {best['quality'].get('wer')}",
        f"- CER: {best['quality'].get('cer')}",
        "",
        "## Hardware Observed",
        "",
        f"- GPU: {best['hardware'].get('gpu_name')}",
        f"- VRAM: {best['hardware'].get('vram_total_mb', 0) / 1024:.1f} GiB",
        f"- CPU count: {best['hardware'].get('cpu_count')}",
        f"- RAM: {best['hardware'].get('ram_total_gib', 0):.1f} GiB",
        f"- Peak measured VRAM: {best.get('telemetry', {}).get('peak_vram_mb')} MB",
        f"- Peak GPU utilization: {best.get('telemetry', {}).get('peak_gpu_util_pct')}%",
        "",
        "## Pipeline Stages",
        "",
        f"- Enabled stages: {', '.join(plan['pipeline']['enabled_stages']) or 'none'}",
        f"- Pipeline latency multiplier: {plan['pipeline']['latency_multiplier']:.2f}",
        f"- Pipeline cost multiplier: {plan['pipeline']['cost_multiplier']:.2f}",
        "",
        "## Capacity And Cost",
        "",
    ]
    for item in plan["hardware_plans"]:
        lines.extend(
            [
                f"### {item['hardware']}",
                "",
                f"- Estimated speed multiplier: {item['estimated_speed_multiplier']:.2f}x",
                f"- Streams/GPU at target utilization: {item['streams_per_gpu_at_target_utilization']:.2f}",
                f"- GPU-hour cost: {money(item['gpu_hour_cost_usd'])}",
                f"- Cost per audio hour: {money(item['cost_per_audio_hour_usd'])}",
                "",
                "| Concurrent streams | GPUs | Servers | Monthly cost | Annual cost |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for scenario in item["scenarios"].values():
            lines.append(
                f"| {scenario['target_concurrent_streams']:,} | {scenario['required_gpus']:,} | "
                f"{scenario['required_servers']:,} | {money(scenario['monthly_cost_usd'])} | {money(scenario['annual_cost_usd'])} |"
            )
        lines.append("")
    best_cost = min(plan["hardware_plans"], key=lambda x: x["cost_per_audio_hour_usd"])
    best_perf = max(plan["hardware_plans"], key=lambda x: x["estimated_speed_multiplier"])
    lines.extend(
        [
            "## Recommendations",
            "",
            f"- Best measured engine/precision from available runs: `{best['engine']}` / `{best['precision']}`.",
            f"- Best projected cost per audio hour in this config: `{best_cost['hardware']}` at {money(best_cost['cost_per_audio_hour_usd'])}.",
            f"- Highest projected single-GPU throughput: `{best_perf['hardware']}`.",
            "- For production, prefer faster-whisper/CTranslate2 with fp16 or int8_float16 after converting fine-tuned checkpoints.",
            "- Use offline batching for backlogs and bounded micro-batches for streaming; keep streaming chunk sizes at 5-10s unless latency requirements force 1-2s chunks.",
            "- Re-run the full matrix before purchase decisions; this report uses measured local data plus editable relative hardware multipliers.",
            "- Whisper large-v3 is feasible when accuracy is the primary requirement, but distillation or smaller Whisper variants should be tested for high-concurrency captioning.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-glob", default=str(ROOT / "benchmark" / "results" / "*.json"))
    p.add_argument("--hardware-costs", default=str(DEFAULT_COSTS))
    p.add_argument("--target-utilization", type=float, default=None)
    p.add_argument("--redundancy-factor", type=float, default=None)
    p.add_argument("--cost-mode", choices=["on_demand", "reserved", "self_hosted_amortized"], default=None)
    p.add_argument("--scenarios", nargs="*", type=int, default=None)
    p.add_argument("--output-json", default=str(DEFAULT_JSON))
    p.add_argument("--output-md", default=str(DEFAULT_REPORT))
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.hardware_costs).read_text())
    defaults = cfg.get("defaults", {})
    utilization = args.target_utilization if args.target_utilization is not None else float(defaults.get("target_utilization", 0.7))
    redundancy = args.redundancy_factor if args.redundancy_factor is not None else float(defaults.get("redundancy_factor", 1.25))
    cost_mode = args.cost_mode or defaults.get("cost_mode", "on_demand")
    scenarios = args.scenarios or list(defaults.get("scenarios", [100, 1000, 10000, 100000]))

    results = load_results(args.results_glob)
    best = best_result(results)
    latency_mult, cost_mult, stages = pipeline_multiplier(cfg)
    measured_speed = float(best["performance"]["speed_multiplier"])
    measured_hph = float(best["performance"]["throughput_audio_hours_per_hour"])
    hw_plans = [
        plan_for_hardware(measured_speed, measured_hph, name, spec, scenarios, utilization, redundancy, cost_mode, latency_mult, cost_mult)
        for name, spec in cfg.get("hardware", {}).items()
    ]
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_count": len(results),
        "target_utilization": utilization,
        "redundancy_factor": redundancy,
        "cost_mode": cost_mode,
        "pipeline": {"latency_multiplier": latency_mult, "cost_multiplier": cost_mult, "enabled_stages": stages},
        "best_benchmark": best,
        "hardware_plans": hw_plans,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    write_markdown(Path(args.output_md), plan)
    print(json.dumps({"capacity_json": str(out_json), "capacity_report": args.output_md, "best_speed_multiplier": measured_speed}, indent=2))


if __name__ == "__main__":
    main()
