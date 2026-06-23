#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "usc": {
        "hf_id": "issai/Uzbek_Speech_Corpus",
        "tier": "gold",
        "notes": "Already staged locally in this project; use only if rebuilding.",
    },
    "common_voice_uz": {
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "config": "uz",
        "tier": "gold",
        "notes": "May require Hugging Face auth and dataset terms acceptance.",
    },
    "fleurs_uz": {
        "hf_id": "google/fleurs",
        "config": "uz_uz",
        "tier": "gold",
        "notes": "Small Uzbek FLEURS subset.",
    },
    "feruza_speech": {
        "hf_id": "REPLACE_WITH_FERUZA_HF_ID",
        "tier": "gold",
        "notes": "Placeholder; set exact HF id or local source before execution.",
    },
    "uzbekvoice_filtered": {
        "hf_id": "REPLACE_WITH_UZBEKVOICE_FILTERED_HF_ID",
        "tier": "silver",
        "notes": "Placeholder; must be filtered before training.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download staged Hugging Face speech datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-root", default="/home/mahmud/datasets")
    parser.add_argument("--execute", action="store_true", help="Actually download. Omit for dry run.")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    output_dir = Path(args.output_root).expanduser() / args.dataset
    print(json.dumps({"dataset": args.dataset, "output_dir": str(output_dir), **spec}, indent=2))

    if not args.execute:
        print("DRY_RUN: add --execute to download.")
        return

    if str(spec["hf_id"]).startswith("REPLACE_WITH_"):
        raise SystemExit(f"Dataset {args.dataset} needs an exact source id before download.")

    from datasets import load_dataset

    kwargs = {}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["hf_id"], **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir / "hf_dataset"))
    (output_dir / "dataset_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"SAVED: {output_dir / 'hf_dataset'}")


if __name__ == "__main__":
    main()

