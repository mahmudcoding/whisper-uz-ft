from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_loader import clean_dataset, load_dataset_flexible, split_dataset


def select_hours(frame: pd.DataFrame, target_hours: float, seed: int = 42) -> pd.DataFrame:
    if frame.empty:
        return frame
    target_seconds = target_hours * 3600.0
    rng = seed
    work = frame.copy()
    if "speaker_id" in work.columns:
        work["_speaker_rank"] = work.groupby("speaker_id")["duration"].transform("count")
        work = work.sample(frac=1.0, random_state=rng).sort_values(["_speaker_rank", "duration"])
    else:
        bins = pd.qcut(work["duration"], q=min(10, max(2, len(work) // 20)), duplicates="drop", labels=False)
        work["_duration_bin"] = bins
        work = work.groupby("_duration_bin", group_keys=False).apply(
            lambda g: g.sample(frac=1.0, random_state=rng), include_groups=False
        )
    rows = []
    total = 0.0
    for row in work.to_dict("records"):
        rows.append(row)
        total += float(row["duration"])
        if total >= target_seconds:
            break
    if not rows:
        return frame.head(0).copy()
    return pd.DataFrame(rows).drop(columns=["_speaker_rank", "_duration_bin"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create representative mini train/val/test splits.")
    parser.add_argument("--dataset-dir", default=str(Path.home() / "datasets/usc"))
    parser.add_argument("--out-dir", default=str(Path.home() / "whisper-uz-ft/data"))
    parser.add_argument("--report", default=str(Path.home() / "whisper-uz-ft/logs/mini_split_report.json"))
    parser.add_argument("--train-hours", type=float, default=2.0)
    parser.add_argument("--val-hours", type=float, default=20.0 / 60.0)
    parser.add_argument("--test-hours", type=float, default=20.0 / 60.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    report_path = Path(args.report).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        report = {"dataset_dir": str(dataset_dir), "error": "dataset directory missing"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    raw = load_dataset_flexible(dataset_dir)
    clean, clean_report = clean_dataset(raw)
    train, val, test, strategy = split_dataset(clean, seed=args.seed)
    mini_train = select_hours(train, args.train_hours, seed=args.seed)
    mini_val = select_hours(val, args.val_hours, seed=args.seed + 1)
    mini_test = select_hours(test, args.test_hours, seed=args.seed + 2)

    keep_cols = [c for c in ("audio_path", "text", "duration", "speaker_id", "split", "source_metadata") if c in clean.columns]
    train[keep_cols].to_csv(out_dir / "train.csv", index=False)
    val[keep_cols].to_csv(out_dir / "val.csv", index=False)
    test[keep_cols].to_csv(out_dir / "test.csv", index=False)
    mini_train[keep_cols].to_csv(out_dir / "mini_train.csv", index=False)
    mini_val[keep_cols].to_csv(out_dir / "mini_val.csv", index=False)
    mini_test[keep_cols].to_csv(out_dir / "mini_test.csv", index=False)

    report = {
        "dataset_dir": str(dataset_dir),
        "cleaning": clean_report,
        "split_strategy": strategy,
        "full_splits": {"train": len(train), "val": len(val), "test": len(test)},
        "mini_splits": {"train": len(mini_train), "val": len(mini_val), "test": len(mini_test)},
        "mini_hours": {
            "train": float(mini_train["duration"].sum() / 3600.0),
            "val": float(mini_val["duration"].sum() / 3600.0),
            "test": float(mini_test["duration"].sum() / 3600.0),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
