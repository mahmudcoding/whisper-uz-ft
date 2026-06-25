#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
RUNNER = PROJECT_ROOT / "scripts/lr_search/run_experiment.py"
AUDITOR = PROJECT_ROOT / "scripts/lr_search/audit_data_leakage.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs_lr_search"
REPORT_ROOT = PROJECT_ROOT / "reports/lr_search"
CONTROLLER_LOG = REPORT_ROOT / "autonomous_search.log"

WER_TIE = 0.003  # 0.3 absolute percentage points when WER is represented as a ratio.
CER_TIE = 0.001  # 0.1 absolute percentage points when CER is represented as a ratio.
DECODER_ONLY_BATCH_OVERRIDES = {
    "per_device_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 16,
}
ENCODER_TUNING_BATCH_OVERRIDES = {
    "per_device_batch_size": 1,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 32,
}


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_test_hashes() -> dict[str, str]:
    audit = json.loads((REPORT_ROOT / "data_leakage_audit.json").read_text(encoding="utf-8"))
    return {
        subset["root"]: subset["manifests"]["test"]["sha256"]
        for subset in audit["subsets"]
    }


def verify_test_hashes(expected: dict[str, str]) -> None:
    for root, expected_hash in expected.items():
        path = Path(root) / "test.csv"
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f"Test manifest changed: {path}; expected {expected_hash}, got {actual}")


def run_audit(expected_hashes: dict[str, str]) -> None:
    subprocess.run([str(PYTHON), str(AUDITOR)], cwd=PROJECT_ROOT, check=True)
    verify_test_hashes(expected_hashes)


def load_metrics(experiment_id: str) -> dict[str, Any] | None:
    path = OUTPUT_ROOT / experiment_id / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def archive_incomplete(output_dir: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = PROJECT_ROOT / "archive/lr_search_incomplete" / f"{output_dir.name}_{stamp}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_dir), str(archive))
    log(f"Archived incomplete run {output_dir} to {archive}")


def execute_experiment(
    config: str,
    experiment_id: str,
    expected_hashes: dict[str, str],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = load_metrics(experiment_id)
    if existing and existing.get("status") == "completed":
        log(f"Reusing completed experiment {experiment_id}")
        verify_test_hashes(expected_hashes)
        return existing

    output_dir = OUTPUT_ROOT / experiment_id
    command = [
        str(PYTHON),
        str(RUNNER),
        "--config",
        str(PROJECT_ROOT / config),
        "--experiment-id",
        experiment_id,
    ]
    if overrides:
        for key, value in overrides.items():
            command.extend(["--set", f"{key}={value}"])

    if output_dir.exists() and any(output_dir.iterdir()):
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            command.extend(["--resume", "auto"])
            log(f"Resuming {experiment_id} from latest checkpoint")
        else:
            archive_incomplete(output_dir)

    log(f"Launching {experiment_id}: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    verify_test_hashes(expected_hashes)
    metrics = load_metrics(experiment_id)
    if result.returncode != 0 or not metrics:
        # One automatic resume attempt is allowed when a valid checkpoint exists.
        checkpoints = sorted((OUTPUT_ROOT / experiment_id).glob("checkpoint-*"))
        if checkpoints:
            log(f"{experiment_id} exited {result.returncode}; attempting one automatic resume")
            retry = command
            if "--resume" not in retry:
                retry = [*retry, "--resume", "auto"]
            retry_result = subprocess.run(retry, cwd=PROJECT_ROOT)
            verify_test_hashes(expected_hashes)
            metrics = load_metrics(experiment_id)
            if retry_result.returncode == 0 and metrics:
                return metrics
        raise RuntimeError(f"Experiment {experiment_id} failed without a recoverable completed result")
    return metrics


def best_validation(metrics: dict[str, Any]) -> dict[str, float] | None:
    value = metrics.get("best_validation_metrics")
    if not value or value.get("eval_wer") is None:
        return None
    return value


def stability_assessment(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not metrics.get("stable"):
        reasons.append("runner marked run unstable")
    validation = best_validation(metrics)
    if validation is None:
        reasons.append("no validation metrics")
    else:
        for key in ("eval_wer", "eval_cer", "eval_loss"):
            value = validation.get(key)
            if value is not None and not math.isfinite(float(value)):
                reasons.append(f"non-finite {key}")
        if float(validation.get("eval_hallucination_rate") or 0.0) >= 0.10:
            reasons.append("extreme hallucination rate")
        if float(validation.get("eval_language_confusion_rate") or 0.0) >= 0.10:
            reasons.append("extreme language confusion")

    losses = [float(item["loss"]) for item in metrics.get("train_loss_curve", []) if item.get("loss") is not None]
    if any(not math.isfinite(value) for value in losses):
        reasons.append("non-finite training loss")
    if len(losses) >= 6:
        first = statistics.median(losses[:3])
        last = statistics.median(losses[-3:])
        if last > max(first * 1.5, first + 5.0):
            reasons.append(f"loss divergence: first median={first:.4f}, last median={last:.4f}")
    return not reasons, reasons


def ranking_key(metrics: dict[str, Any]) -> tuple[float, float, int]:
    validation = best_validation(metrics) or {}
    stable, _ = stability_assessment(metrics)
    return (
        float(validation.get("eval_wer", float("inf"))),
        float(validation.get("eval_cer", float("inf"))),
        0 if stable else 1,
    )


def format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def report_table(title: str, rows: list[dict[str, Any]], path: Path, narrative: list[str]) -> None:
    lines = [
        f"# {title}",
        "",
        *narrative,
        "",
        "| Experiment | Decoder LR | Encoder LR | Mode | WER | CER | Eval loss | Hallucination | Stable | Decision |",
        "|---|---:|---:|---|---:|---:|---:|---:|:---:|---|",
    ]
    for item in rows:
        metrics = item["metrics"]
        validation = best_validation(metrics) or {}
        stable, reasons = stability_assessment(metrics)
        lines.append(
            f"| {metrics['experiment_id']} | {metrics.get('decoder_learning_rate')} | "
            f"{metrics.get('encoder_learning_rate')} | {metrics.get('tuning_mode')} | "
            f"{format_metric(validation.get('eval_wer'))} | {format_metric(validation.get('eval_cer'))} | "
            f"{format_metric(validation.get('eval_loss'))} | "
            f"{format_metric(validation.get('eval_hallucination_rate'))} | "
            f"{'yes' if stable else 'no'} | {item.get('decision') or '; '.join(reasons)} |"
        )
    lines.extend(["", *narrative, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def phase1a(expected_hashes: dict[str, str]) -> list[tuple[str, float]]:
    candidates = [
        ("configs/lr_search/phase1a_decoder_lr_2e6.yaml", "phase1a_decoder_lr_2e6", 2e-6),
        ("configs/lr_search/phase1a_decoder_lr_8e6.yaml", "phase1a_decoder_lr_8e6", 8e-6),
        ("configs/lr_search/phase1a_decoder_lr_2e5.yaml", "phase1a_decoder_lr_2e5", 2e-5),
        ("configs/lr_search/phase1a_decoder_lr_5e5.yaml", "phase1a_decoder_lr_5e5", 5e-5),
    ]
    rows = []
    survivors = []
    for config, experiment_id, lr in candidates:
        metrics = execute_experiment(config, experiment_id, expected_hashes)
        stable, reasons = stability_assessment(metrics)
        decision = "SURVIVE" if stable else f"REJECT: {'; '.join(reasons)}"
        rows.append({"metrics": metrics, "decision": decision})
        if stable:
            survivors.append((config.replace("phase1a_", ""), lr))
    report_table(
        "Phase 1A Divergence Screen",
        rows,
        REPORT_ROOT / "phase1a_divergence_screen.md",
        [
            "Each decoder-only candidate ran for 300 optimizer steps on `coarse_10h`.",
            f"Survivors: `{[lr for _, lr in survivors]}`.",
        ],
    )
    if not survivors:
        raise RuntimeError("All Phase 1A candidates were unstable; redesign is required")
    return survivors


def phase1b(
    survivors: list[tuple[str, float]],
    expected_hashes: dict[str, str],
) -> list[tuple[float, dict[str, Any]]]:
    results = []
    rows = []
    for config, lr in survivors:
        label = Path(config).stem
        experiment_id = f"phase1b_{label}"
        metrics = execute_experiment(
            config,
            experiment_id,
            expected_hashes,
            DECODER_ONLY_BATCH_OVERRIDES,
        )
        stable, reasons = stability_assessment(metrics)
        rows.append(
            {
                "metrics": metrics,
                "decision": "ELIGIBLE" if stable else f"REJECT: {'; '.join(reasons)}",
            }
        )
        if stable:
            results.append((lr, metrics))
    results.sort(key=lambda item: ranking_key(item[1]))
    promoted = results[:2]
    for row in rows:
        lr = float(row["metrics"]["decoder_learning_rate"])
        if any(math.isclose(lr, promoted_lr) for promoted_lr, _ in promoted):
            row["decision"] = "PROMOTE TO PHASE 2"

    tie_note = "Fewer than two stable candidates survived."
    if len(promoted) >= 2:
        first = best_validation(promoted[0][1]) or {}
        second = best_validation(promoted[1][1]) or {}
        wer_delta = abs(float(first["eval_wer"]) - float(second["eval_wer"]))
        cer_delta = abs(float(first["eval_cer"]) - float(second["eval_cer"]))
        tie_note = (
            f"Top-two deltas: WER `{wer_delta:.6f}`, CER `{cer_delta:.6f}`. "
            f"Statistical tie thresholds are WER `{WER_TIE}` and CER `{CER_TIE}`. "
            f"Tied: `{'yes' if wer_delta < WER_TIE or cer_delta < CER_TIE else 'no'}`."
        )
    report_table(
        "Phase 1B Decoder Results",
        rows,
        REPORT_ROOT / "phase1b_decoder_results.md",
        [
            "Full coarse runs use validation metrics only. The top two stable candidates are promoted.",
            tie_note,
        ],
    )
    if not promoted:
        raise RuntimeError("No stable Phase 1B candidates")
    return promoted


def choose_decoder(results: list[tuple[float, dict[str, Any]]]) -> tuple[float, dict[str, Any], str]:
    ranked = sorted(results, key=lambda item: ranking_key(item[1]))
    winner = ranked[0]
    reasoning = "Lowest validation WER."
    if len(ranked) > 1:
        first_val = best_validation(ranked[0][1]) or {}
        second_val = best_validation(ranked[1][1]) or {}
        wer_delta = abs(float(first_val["eval_wer"]) - float(second_val["eval_wer"]))
        cer_delta = abs(float(first_val["eval_cer"]) - float(second_val["eval_cer"]))
        if wer_delta < WER_TIE:
            if cer_delta >= CER_TIE:
                winner = min(ranked[:2], key=lambda item: float((best_validation(item[1]) or {})["eval_cer"]))
                reasoning = (
                    f"WER difference `{wer_delta:.6f}` is within noise threshold; "
                    f"selected lower CER because CER difference `{cer_delta:.6f}` is material."
                )
            else:
                winner = min(ranked[:2], key=lambda item: item[0])
                reasoning = (
                    f"WER `{wer_delta:.6f}` and CER `{cer_delta:.6f}` differences are statistical ties; "
                    "selected the lower LR as the more conservative generalization choice."
                )
    return winner[0], winner[1], reasoning


def phase2_decoders(
    promoted: list[tuple[float, dict[str, Any]]],
    expected_hashes: dict[str, str],
) -> tuple[float, dict[str, Any]]:
    results = []
    rows = []
    for lr, _ in promoted:
        label = f"{lr:.0e}".replace("-", "m")
        metrics = execute_experiment(
            "configs/lr_search/decoder_best_main.yaml",
            f"phase2_decoder_{label}",
            expected_hashes,
            {
                **DECODER_ONLY_BATCH_OVERRIDES,
                "decoder_learning_rate": lr,
                "learning_rate": lr,
            },
        )
        results.append((lr, metrics))
        rows.append({"metrics": metrics, "decision": "CANDIDATE"})
    winner_lr, winner_metrics, reasoning = choose_decoder(results)
    for row in rows:
        if math.isclose(float(row["metrics"]["decoder_learning_rate"]), winner_lr):
            row["decision"] = "SELECTED"
    report_table(
        "Phase 2 Decoder Confirmation",
        rows,
        REPORT_ROOT / "phase2_decoder_results.md",
        [
            "Candidates were evaluated on the 30h proxy. Tiny differences are treated as noise.",
            f"Decision: decoder LR `{winner_lr}`. {reasoning}",
        ],
    )
    return winner_lr, winner_metrics


def phase2_upper_encoder(
    decoder_lr: float,
    decoder_metrics: dict[str, Any],
    expected_hashes: dict[str, str],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        ("configs/lr_search/upper_encoder_lr_5e7.yaml", 5e-7),
        ("configs/lr_search/upper_encoder_lr_1e6.yaml", 1e-6),
        ("configs/lr_search/upper_encoder_lr_2e6.yaml", 2e-6),
        ("configs/lr_search/upper_encoder_lr_5e6.yaml", 5e-6),
    ]
    all_results: list[tuple[float | None, dict[str, Any]]] = [(None, decoder_metrics)]
    rows = [{"metrics": decoder_metrics, "decision": "DECODER-ONLY BASELINE"}]
    for config, encoder_lr in candidates:
        label = f"{encoder_lr:.0e}".replace("-", "m")
        metrics = execute_experiment(
            config,
            f"phase2_upper_encoder_{label}",
            expected_hashes,
            {
                **ENCODER_TUNING_BATCH_OVERRIDES,
                "decoder_learning_rate": decoder_lr,
                "learning_rate": decoder_lr,
            },
        )
        all_results.append((encoder_lr, metrics))
        rows.append({"metrics": metrics, "decision": "CANDIDATE"})
    stable_results = [item for item in all_results if stability_assessment(item[1])[0]]
    winner = min(stable_results, key=lambda item: ranking_key(item[1]))
    stable_encoder_results = [
        item for item in all_results
        if item[0] is not None and stability_assessment(item[1])[0]
    ]
    if not stable_encoder_results:
        raise RuntimeError("All upper-encoder LR candidates were unstable")
    best_encoder = min(stable_encoder_results, key=lambda item: ranking_key(item[1]))
    for row in rows:
        if row["metrics"]["experiment_id"] == winner[1]["experiment_id"]:
            row["decision"] = "SELECTED REGIME"
    report_table(
        "Phase 2 Upper Encoder Results",
        rows,
        REPORT_ROOT / "phase2_upper_encoder_results.md",
        [
            "Decoder-only is included as the control. Encoder 24-31 is promoted only if validation",
            "evidence beats or materially ties the decoder-only result without stability regression.",
        ],
    )
    return float(best_encoder[0]), best_encoder[1], rows


def phase3_boundary(
    decoder_lr: float,
    best_encoder_lr: float,
    decoder_metrics: dict[str, Any],
    best_upper_metrics: dict[str, Any],
    expected_hashes: dict[str, str],
) -> tuple[str, float | None, dict[str, Any], list[dict[str, Any]]]:
    rows = [
        {"metrics": decoder_metrics, "decision": "DECODER-ONLY CONTROL"},
        {"metrics": best_upper_metrics, "decision": "ENCODER 24-31 CONTROL"},
    ]
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("decoder_only", decoder_metrics),
        ("encoder_24_31_plus_decoder", best_upper_metrics),
    ]
    metrics = execute_experiment(
        "configs/lr_search/freeze_boundary_15.yaml",
        "phase3_freeze_boundary_15",
        expected_hashes,
        {
            **ENCODER_TUNING_BATCH_OVERRIDES,
            "decoder_learning_rate": decoder_lr,
            "learning_rate": decoder_lr,
            "encoder_learning_rate": best_encoder_lr,
        },
    )
    rows.append({"metrics": metrics, "decision": "ENCODER 16-31 CANDIDATE"})
    candidates.append(("encoder_16_31_plus_decoder", metrics))
    stable = [item for item in candidates if stability_assessment(item[1])[0]]
    winner = min(stable, key=lambda item: ranking_key(item[1]))
    for row in rows:
        if row["metrics"]["experiment_id"] == winner[1]["experiment_id"]:
            row["decision"] = "SELECTED"
    report_table(
        "Phase 3 Freeze Boundary Results",
        rows,
        REPORT_ROOT / "phase3_freeze_boundary_results.md",
        [
            "All three requested regimes are compared on the 30h proxy: decoder-only, the best",
            "encoder 24-31 candidate, and encoder 16-31 using the same encoder/decoder LRs.",
        ],
    )
    selected_encoder_lr = None if winner[0] == "decoder_only" else best_encoder_lr
    return winner[0], selected_encoder_lr, winner[1], rows


def write_final(
    decoder_lr: float,
    encoder_lr: float | None,
    regime: str,
    metrics: dict[str, Any],
) -> None:
    validation = best_validation(metrics) or {}
    regime_labels = {
        "decoder_only": "A) decoder only",
        "encoder_24_31_plus_decoder": "B) encoder 24-31 + decoder",
        "encoder_16_31_plus_decoder": "C) encoder 16-31 + decoder",
    }
    confidence = "medium"
    lines = [
        "# Final LR Search Recommendation",
        "",
        "## Selected Configuration",
        "",
        f"- Best decoder LR: `{decoder_lr}`",
        f"- Best upper encoder LR: `{encoder_lr if encoder_lr is not None else 'not applicable; encoder frozen'}`",
        f"- Best freeze boundary: `{regime}`",
        f"- Best regime: **{regime_labels.get(regime, regime)}**",
        f"- Validation WER: `{validation.get('eval_wer')}`",
        f"- Validation CER: `{validation.get('eval_cer')}`",
        f"- Confidence: **{confidence}**",
        "",
        "## Regime Conclusion",
        "",
        f"- A) decoder only: `{'SELECTED' if regime == 'decoder_only' else 'not selected'}`",
        f"- B) encoder 24-31 + decoder: `{'SELECTED' if regime == 'encoder_24_31_plus_decoder' else 'not selected'}`",
        f"- C) encoder 16-31 + decoder: `{'SELECTED' if regime == 'encoder_16_31_plus_decoder' else 'not selected'}`",
        "- D) full FT: **rejected for current data scale**; the measured USC full-FT run",
        "  degraded WER/CER relative to partial FT and provides no evidence for lower-encoder updates.",
        "",
        "## Recommended 207h Gold Training Config",
        "",
        "```yaml",
        "model_name: openai/whisper-large-v3",
        "language: uz",
        "task: transcribe",
        "data_dir: ~/whisper-uz-ft/data/gold_master_training_schema",
        f"tuning_mode: {regime}",
        f"decoder_learning_rate: {decoder_lr}",
        f"encoder_learning_rate: {encoder_lr if encoder_lr is not None else decoder_lr}",
        f"learning_rate: {decoder_lr}",
        "epochs: 1",
        "per_device_batch_size: 1",
        "gradient_accumulation_steps: 32",
        "warmup_ratio: 0.10",
        "weight_decay: 0.01",
        "max_grad_norm: 1.0",
        "scheduler: cosine",
        "bf16: true",
        "fp16: false",
        "gradient_checkpointing: true",
        "metric_for_best_model: wer",
        "greater_is_better: false",
        "generation_num_beams: 1",
        "```",
        "",
        "The final Gold run should preserve a locked test set and use validation-only checkpoint",
        "selection. One epoch is the initial production recommendation; extend only if validation",
        "continues improving without overfitting.",
        "",
    ]
    (REPORT_ROOT / "FINAL_RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    expected_hashes = initial_test_hashes()
    run_audit(expected_hashes)
    log("Leakage audit passed; beginning Phase 1A")
    survivors = phase1a(expected_hashes)
    run_audit(expected_hashes)
    promoted = phase1b(survivors, expected_hashes)
    run_audit(expected_hashes)
    decoder_lr, decoder_metrics = phase2_decoders(promoted, expected_hashes)
    run_audit(expected_hashes)
    best_encoder_lr, best_upper_metrics, _ = phase2_upper_encoder(
        decoder_lr, decoder_metrics, expected_hashes
    )
    run_audit(expected_hashes)
    regime, selected_encoder_lr, final_metrics, _ = phase3_boundary(
        decoder_lr, best_encoder_lr, decoder_metrics, best_upper_metrics, expected_hashes
    )
    verify_test_hashes(expected_hashes)
    write_final(decoder_lr, selected_encoder_lr, regime, final_metrics)
    subprocess.run(
        [str(PYTHON), str(PROJECT_ROOT / "scripts/lr_search/compare_experiments.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    log("Autonomous LR search completed; FINAL_RECOMMENDATION.md generated")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        raise
