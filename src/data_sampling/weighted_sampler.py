from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_TIER_WEIGHTS = {
    "gold": 4.0,
    "silver": 1.5,
    "bronze": 1.0,
}


def sample_weight(row: dict, dataset_weights: dict[str, float] | None = None) -> float:
    tier = str(row.get("quality_class") or row.get("tier") or "bronze").lower()
    base = DEFAULT_TIER_WEIGHTS.get(tier, 1.0)
    dataset_id = str(row.get("dataset_id") or "")
    if dataset_weights and dataset_id in dataset_weights:
        base *= float(dataset_weights[dataset_id])
    quality = row.get("quality_score")
    if quality not in (None, ""):
        base *= max(0.1, min(1.25, float(quality) / 100.0))
    return float(base)


def add_sampling_weights(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as f_in, output_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = list(dict.fromkeys(list(reader.fieldnames or []) + ["sampling_weight"]))
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            row["sampling_weight"] = sample_weight(row)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    add_sampling_weights(Path(args.input_csv), Path(args.output_csv))


if __name__ == "__main__":
    main()

