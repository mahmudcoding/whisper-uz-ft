from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(Path.home() / "whisper-uz-ft/logs/gpu_report.txt"))
    args = parser.parse_args()
    report = Path(args.report).expanduser()
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"torch_version: {torch.__version__}")
    lines.append(f"torch_cuda_available: {torch.cuda.is_available()}")
    lines.append(f"torch_cuda_version: {torch.version.cuda}")
    lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        lines.append(f"gpu_name: {props.name}")
        lines.append(f"vram_gib: {props.total_memory / 1024**3:.2f}")
        lines.append(f"fp16_available: {props.major >= 7}")
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        y = x @ x
        lines.append(f"fp16_matmul_ok: {torch.isfinite(y).all().item()}")
        lines.append(f"allocated_after_test_mib: {torch.cuda.memory_allocated() / 1024**2:.2f}")
    try:
        import bitsandbytes as bnb

        lines.append(f"bitsandbytes_status: ok {getattr(bnb, '__version__', 'unknown')}")
    except Exception as exc:
        lines.append(f"bitsandbytes_status: failed {exc!r}")
    try:
        import accelerate
        import transformers

        lines.append(f"transformers_version: {transformers.__version__}")
        lines.append(f"accelerate_version: {accelerate.__version__}")
    except Exception as exc:
        lines.append(f"hf_stack_status: failed {exc!r}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
