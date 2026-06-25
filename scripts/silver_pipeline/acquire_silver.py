#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumably download pinned SILVER dataset snapshots.")
    parser.add_argument("--config", default=str(ROOT / "configs/silver_datasets.yaml"))
    parser.add_argument("--dataset", action="append", help="Dataset key; repeat to limit acquisition.")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(cfg["output_root"]).expanduser()
    selected = set(args.dataset or cfg["datasets"].keys())
    records = []
    for name, spec in cfg["datasets"].items():
        if name not in selected:
            continue
        destination = output_root / name / "raw_hf"
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(ROOT / ".venv/bin/hf"),
            "download",
            spec["hf_id"],
            "--repo-type",
            "dataset",
            "--revision",
            spec["revision"],
            "--local-dir",
            str(destination),
            "--max-workers",
            str(args.max_workers),
        ]
        started = datetime.now(timezone.utc)
        print(f"ACQUIRE {name}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        record = {
            "dataset": name,
            "hf_id": spec["hf_id"],
            "revision": spec["revision"],
            "destination": str(destination),
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
        }
        records.append(record)
        (output_root / name / "acquisition.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        if completed.returncode:
            raise SystemExit(f"Acquisition failed for {name}: exit {completed.returncode}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
