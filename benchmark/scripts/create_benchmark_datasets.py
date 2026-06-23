from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


GROUPS = {
    "clean_read_speech": {"target_hours": 1.0, "duration_min": 2.0, "duration_max": 12.0},
    "meeting_audio": {"target_hours": 1.0, "duration_min": 4.0, "duration_max": 30.0},
    "noisy_audio": {"target_hours": 1.0, "duration_min": 1.0, "duration_max": 30.0},
    "long_recordings": {"target_hours": 1.0, "duration_min": 5.0, "duration_max": 30.0},
    "smoke": {"target_hours": 0.03, "duration_min": 1.0, "duration_max": 12.0},
}


def select_duration(df: pd.DataFrame, target_hours: float, seed: int, duration_min: float, duration_max: float) -> pd.DataFrame:
    work = df[(df["duration"] >= duration_min) & (df["duration"] <= duration_max)].copy()
    if work.empty:
        work = df.copy()
    if "speaker_id" in work.columns:
        work = work.sample(frac=1.0, random_state=seed).sort_values(["speaker_id", "duration"])
        work = work.groupby("speaker_id", group_keys=False).head(max(1, math.ceil(len(work) / max(1, work["speaker_id"].nunique()))))
        work = work.sample(frac=1.0, random_state=seed + 11)
    else:
        work = work.sample(frac=1.0, random_state=seed)
    target_seconds = target_hours * 3600.0
    rows = []
    total = 0.0
    for row in work.to_dict("records"):
        rows.append(row)
        total += float(row["duration"])
        if total >= target_seconds:
            break
    return pd.DataFrame(rows)


def synthesize_long_recordings(df: pd.DataFrame, out_dir: Path, target_hours: float) -> pd.DataFrame:
    """Create manifest entries for long-mode by grouping short USC files.

    The benchmark runner treats rows with `segments` as virtual long recordings and
    transcribes the segments sequentially. This avoids destructive audio concatenation
    and still measures long-job scheduling behavior.
    """
    work = df.sample(frac=1.0, random_state=202).copy()
    target = target_hours * 3600.0
    groups = []
    current = []
    seconds = 0.0
    group_id = 0
    for row in work.to_dict("records"):
        current.append(row)
        seconds += float(row["duration"])
        if seconds >= min(1800.0, target):
            groups.append(
                {
                    "audio_path": f"virtual_long_recording_{group_id}",
                    "text": " ".join(str(x["text"]) for x in current),
                    "duration": seconds,
                    "speaker_id": "virtual_multi",
                    "split": "benchmark",
                    "source_metadata": "virtual_concat",
                    "segments": "|".join(str(x["audio_path"]) for x in current),
                }
            )
            group_id += 1
            current = []
            seconds = 0.0
        if sum(float(g["duration"]) for g in groups) >= target:
            break
    return pd.DataFrame(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default="data/test.csv")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--out-dir", default="benchmark/datasets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source_csv)
    train = pd.read_csv(args.train_csv) if Path(args.train_csv).exists() else source
    combined = pd.concat([source, train], ignore_index=True)

    manifests = {}
    for i, (name, cfg) in enumerate(GROUPS.items()):
        if name == "long_recordings":
            selected = synthesize_long_recordings(combined, out_dir, cfg["target_hours"])
        elif name == "noisy_audio":
            # USC does not include noise labels; use longest/hardest utterances as a deterministic proxy.
            selected = combined.sort_values("duration", ascending=False).head(500)
            selected = select_duration(selected, cfg["target_hours"], 100 + i, cfg["duration_min"], cfg["duration_max"])
        elif name == "meeting_audio":
            selected = select_duration(combined, cfg["target_hours"], 100 + i, cfg["duration_min"], cfg["duration_max"])
        else:
            selected = select_duration(source, cfg["target_hours"], 100 + i, cfg["duration_min"], cfg["duration_max"])
        path = out_dir / f"{name}.csv"
        selected.to_csv(path, index=False)
        manifests[name] = {
            "path": str(path),
            "samples": int(len(selected)),
            "hours": float(selected["duration"].sum() / 3600.0) if not selected.empty else 0.0,
        }
    pd.DataFrame.from_dict(manifests, orient="index").to_csv(out_dir / "manifest_summary.csv")
    print(pd.DataFrame.from_dict(manifests, orient="index").to_string())


if __name__ == "__main__":
    main()
