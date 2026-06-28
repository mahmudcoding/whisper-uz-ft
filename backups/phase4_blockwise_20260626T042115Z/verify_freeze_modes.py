#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import detailed_trainable_parameter_report, load_whisper_for_partial_ft


EXPECTED = {
    "decoder_only": ("frozen", "frozen", "frozen", "trainable"),
    "encoder_24_31_plus_decoder": ("frozen", "frozen", "trainable", "trainable"),
    "encoder_16_31_plus_decoder": ("frozen", "mixed", "trainable", "trainable"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Whisper and verify all LR-search freeze modes.")
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--mode", choices=sorted(EXPECTED), action="append")
    parser.add_argument("--output", type=Path, default=Path("reports/lr_search/freeze_mode_verification.json"))
    args = parser.parse_args()

    modes = args.mode or list(EXPECTED)
    reports = {}
    for mode in modes:
        bundle = load_whisper_for_partial_ft(
            args.model,
            language="uz",
            task="transcribe",
            tuning_mode=mode,
            gradient_checkpointing=False,
        )
        report = detailed_trainable_parameter_report(bundle.model)
        actual = tuple(
            report[group]["state"]
            for group in ("encoder_0_7", "encoder_8_23", "encoder_24_31", "decoder")
        )
        report["expected_states"] = EXPECTED[mode]
        report["verification_ok"] = actual == EXPECTED[mode]
        reports[mode] = report
        del bundle
        if not report["verification_ok"]:
            raise RuntimeError(f"{mode}: expected {EXPECTED[mode]}, got {actual}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
