from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from data_loader import AUDIO_EXTS, find_metadata_files, read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect USC dataset structure.")
    parser.add_argument("--dataset-dir", default=str(Path.home() / "datasets/usc"))
    parser.add_argument("--report", default=str(Path.home() / "whisper-uz-ft/logs/dataset_structure_report.txt"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    report = Path(args.report).expanduser()
    report.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"Dataset directory: {dataset_dir}")
    lines.append(f"Exists: {dataset_dir.exists()}")
    if not dataset_dir.exists():
        lines.append("ERROR: dataset directory is missing.")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        raise SystemExit(2)

    files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    dirs = [p for p in dataset_dir.rglob("*") if p.is_dir()]
    ext_counts = Counter(p.suffix.lower() or "[no_ext]" for p in files)
    audio_counts = Counter(p.suffix.lower() for p in files if p.suffix.lower() in AUDIO_EXTS)
    metadata = find_metadata_files(dataset_dir)

    lines.append("")
    lines.append("Directory tree (first 200 paths):")
    for p in list(dataset_dir.rglob("*"))[:200]:
        lines.append(str(p))

    lines.append("")
    lines.append(f"Total files: {len(files)}")
    lines.append(f"Total directories: {len(dirs)}")
    lines.append("")
    lines.append("File extensions:")
    for ext, count in ext_counts.most_common():
        lines.append(f"{ext}\t{count}")

    lines.append("")
    lines.append("Audio file counts by extension:")
    if audio_counts:
        for ext, count in audio_counts.most_common():
            lines.append(f"{ext}\t{count}")
    else:
        lines.append("No supported audio files found.")

    lines.append("")
    lines.append("Metadata files:")
    if metadata:
        for meta in metadata[:100]:
            lines.append(f"{meta}")
            try:
                frame = read_metadata(meta)
                lines.append(f"  rows={len(frame)} columns={list(frame.columns)}")
                lines.append(f"  head={frame.head(3).to_dict('records')}")
            except Exception as exc:
                lines.append(f"  ERROR reading metadata: {exc!r}")
    else:
        lines.append("No csv/tsv/json/jsonl/parquet metadata files found.")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
