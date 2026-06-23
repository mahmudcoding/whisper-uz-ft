#!/usr/bin/env python3
"""Generate a long-form offline capacity report from benchmark JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def money(v: float) -> str:
    return f"${v:,.4f}"


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def flatten(results: list[dict[str, Any]], gpu_hour_cost: float, target_util: float) -> pd.DataFrame:
    rows = []
    for d in results:
        perf = d["performance"]
        timing = d["timing"]
        tele = d["telemetry"]
        audio_hph = float(perf["throughput_audio_hours_per_hour"])
        rows.append(
            {
                "batch_size": int(d["batch_size"]),
                "beam_size": int(d["beam_size"]),
                "audio_hours": float(timing["total_audio_seconds"]) / 3600.0,
                "startup_seconds": float(timing["startup_overhead_seconds"]),
                "processing_seconds": float(timing["processing_seconds"]),
                "total_wall_seconds": float(timing["total_wall_seconds"]),
                "rtf_processing": float(perf["rtf"]),
                "rtf_end_to_end": float(timing["total_wall_seconds"]) / float(timing["total_audio_seconds"]),
                "speed_multiplier": float(perf["speed_multiplier"]),
                "throughput_audio_hours_per_hour": audio_hph,
                "cost_per_audio_hour_usd": gpu_hour_cost / max(audio_hph * target_util, 1e-9),
                "avg_latency_seconds": timing.get("avg_latency_seconds"),
                "p95_latency_seconds": timing.get("p95_latency_seconds"),
                "avg_gpu_util_pct": tele.get("avg_gpu_util_pct"),
                "peak_gpu_util_pct": tele.get("peak_gpu_util_pct"),
                "avg_vram_mb": tele.get("avg_vram_mb"),
                "peak_vram_mb": tele.get("peak_vram_mb"),
                "avg_cpu_percent": tele.get("avg_cpu_percent"),
                "peak_cpu_percent": tele.get("peak_cpu_percent"),
                "wer": d["quality"].get("wer"),
                "cer": d["quality"].get("cer"),
                "result_json": d["_path"],
            }
        )
    return pd.DataFrame(rows).sort_values(["beam_size", "batch_size"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-glob", default="benchmark/results/long_form_offline_faster_whisper_large-v3_fp16_bs*_beam*.json")
    p.add_argument("--hardware-costs", default="benchmark/configs/hardware_costs.yaml")
    p.add_argument("--output-md", default="benchmark/reports/long_form_offline_capacity_report.md")
    p.add_argument("--output-csv", default="benchmark/reports/long_form_offline_capacity_report.csv")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.hardware_costs).read_text())
    defaults = cfg.get("defaults", {})
    target_util = float(defaults.get("target_utilization", 0.70))
    cost_mode = defaults.get("cost_mode", "on_demand")
    a40 = cfg["hardware"]["A40"]
    gpu_hour_cost = float(a40["pricing"][cost_mode])
    scenarios = list(defaults.get("scenarios", [100, 1000, 10000, 100000]))

    paths = [Path(p) for p in sorted(glob.glob(args.results_glob))]
    if not paths:
        raise FileNotFoundError(args.results_glob)
    results = [load_result(p) for p in paths]
    df = flatten(results, gpu_hour_cost, target_util)
    best = df.sort_values("throughput_audio_hours_per_hour", ascending=False).iloc[0]
    beam1_df = df[df["beam_size"] == 1]
    beam5_df = df[df["beam_size"] == 5]
    beam1_best = beam1_df.sort_values("throughput_audio_hours_per_hour", ascending=False).iloc[0] if not beam1_df.empty else None
    beam5_best = beam5_df.sort_values("throughput_audio_hours_per_hour", ascending=False).iloc[0] if not beam5_df.empty else None
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    lines = [
        "# Long-Form Offline Whisper Capacity Report",
        "",
        "Configuration: `faster-whisper`, model `large-v3`, precision `fp16`, USC-derived 5+ hour long-form offline dataset.",
        "",
        f"Measured runs: {len(df)}",
        f"GPU-hour cost assumption: {money(gpu_hour_cost)} for A40 `{cost_mode}` pricing.",
        f"Target utilization for cost/audio-hour: {target_util:.2f}",
        "",
        "## Best Throughput Result",
        "",
        f"- Batch size: `{int(best['batch_size'])}`",
        f"- Beam size: `{int(best['beam_size'])}`",
        f"- End-to-end RTF: `{best['rtf_end_to_end']:.4f}`",
        f"- Processing RTF: `{best['rtf_processing']:.4f}`",
        f"- Throughput: `{best['throughput_audio_hours_per_hour']:.2f}` audio-hours/hour/GPU",
        f"- Cost/audio-hour: `{money(best['cost_per_audio_hour_usd'])}`",
        f"- Peak VRAM: `{best['peak_vram_mb']:.0f}` MB",
        f"- Average GPU utilization: `{best['avg_gpu_util_pct']:.1f}%`",
        "",
        "## Batch And Beam Matrix",
        "",
        "| Beam | Batch | Audio h | Startup s | Wall s | RTF e2e | RTF proc | Throughput h/h | Cost/audio h | Avg GPU % | Peak GPU % | Peak VRAM MB | Avg CPU % | P95 latency s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['beam_size'])} | {int(r['batch_size'])} | {r['audio_hours']:.2f} | {r['startup_seconds']:.1f} | "
            f"{r['total_wall_seconds']:.1f} | {r['rtf_end_to_end']:.4f} | {r['rtf_processing']:.4f} | "
            f"{r['throughput_audio_hours_per_hour']:.2f} | {money(r['cost_per_audio_hour_usd'])} | "
            f"{r['avg_gpu_util_pct']:.1f} | {r['peak_gpu_util_pct']:.1f} | {r['peak_vram_mb']:.0f} | "
            f"{r['avg_cpu_percent']:.1f} | {r['p95_latency_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Offline Capacity Numbers",
            "",
            (
                f"- Best beam=1 batch: batch `{int(beam1_best['batch_size'])}`, "
                f"`{beam1_best['throughput_audio_hours_per_hour']:.2f}` audio-hours/hour/GPU."
                if beam1_best is not None
                else "- Beam=1 was not measured."
            ),
            (
                f"- Best beam=5 batch: batch `{int(beam5_best['batch_size'])}`, "
                f"`{beam5_best['throughput_audio_hours_per_hour']:.2f}` audio-hours/hour/GPU."
                if beam5_best is not None
                else "- Beam=5 was skipped/aborted for this report; no completed full-duration beam=5 result is included."
            ),
            f"- Startup overhead is `{best['startup_seconds']:.1f}` seconds on the best run, amortized over `{best['audio_hours']:.2f}` audio hours.",
            "",
            "| Concurrent real-time equivalent streams | Required A40 GPUs at best measured throughput |",
            "|---:|---:|",
        ]
    )
    streams_per_gpu = float(best["speed_multiplier"]) * target_util
    for streams in scenarios:
        lines.append(f"| {streams:,} | {math.ceil(streams / max(streams_per_gpu, 1e-9)):,} |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"For long-form offline throughput on this A40, use batch size `{int(best['batch_size'])}` with beam size `{int(best['beam_size'])}` when throughput is the priority.",
            "Use beam size 5 only with decode guardrails such as VAD, max token limits, and hallucination controls; the attempted unguarded beam=5 run was skipped after sustained GPU-bound decoding.",
            f"The raw CSV backing this report is `{args.output_csv}`.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(args.output_md)
    print(args.output_csv)
    print(json.dumps({"best_batch_size": int(best["batch_size"]), "best_beam_size": int(best["beam_size"]), "best_throughput_hph": float(best["throughput_audio_hours_per_hour"]), "best_cost_per_audio_hour_usd": float(best["cost_per_audio_hour_usd"])}, indent=2))


if __name__ == "__main__":
    main()
