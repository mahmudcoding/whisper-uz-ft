from __future__ import annotations

import argparse
from pathlib import Path

from model import load_whisper_for_partial_ft, trainable_parameter_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--report", default=str(Path.home() / "whisper-uz-ft/logs/model_trainable_report.txt"))
    args = parser.parse_args()
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_whisper_for_partial_ft(args.model, train_last_encoder_blocks=8, gradient_checkpointing=True)
    model = bundle.model
    encoder_layers = model.model.encoder.layers
    decoder = model.model.decoder
    report = trainable_parameter_report(model)
    total = report["total_parameters"]
    trainable = report["trainable_parameters"]
    frozen = total - trainable

    lines = [
        f"model: {args.model}",
        f"encoder_layer_count: {len(encoder_layers)}",
        f"expected_frozen_encoder_blocks: 0-23",
        f"expected_trainable_encoder_blocks: 24-31",
        f"decoder_expected_trainable: true",
        f"total_params: {total}",
        f"trainable_params: {trainable}",
        f"frozen_params: {frozen}",
        f"trainable_percent: {report['trainable_percent']:.4f}",
        "",
        "encoder_blocks:",
    ]
    ok = len(encoder_layers) == 32
    for i, layer in enumerate(encoder_layers):
        layer_trainable = any(p.requires_grad for p in layer.parameters())
        expected = i >= 24
        lines.append(f"encoder_block_{i}: trainable={layer_trainable} expected={expected}")
        ok = ok and layer_trainable == expected
    decoder_trainable = all(p.requires_grad for p in decoder.parameters())
    lines.append("")
    lines.append(f"decoder_all_trainable: {decoder_trainable}")
    ok = ok and decoder_trainable
    lines.append(f"freeze_logic_ok: {ok}")
    if not ok:
        raise RuntimeError("Freeze logic verification failed. See report.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
