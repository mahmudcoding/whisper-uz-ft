#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SUMMARY = ROOT / "reports/silver_quality_report/summary.json"
START = "<!-- SILVER_PIPELINE:START -->"
END = "<!-- SILVER_PIPELINE:END -->"


def replace_section(path: Path, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    section = f"{START}\n{content.rstrip()}\n{END}"
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        text = f"{before}\n\n{section}\n\n{after}".rstrip() + "\n"
    else:
        text = text.rstrip() + f"\n\n{section}\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = int(summary["final_unique_rows"])
    hours = float(summary["final_unique_hours"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dataset_lines = "\n".join(
        f"| {name} | {values['rows']:,} | {values['hours']:.2f} | {values['speakers']:,} |"
        for name, values in summary["final_by_dataset"].items()
    )
    section = f"""## SILVER Corpus

Last verified: `{now}`

- Status: complete and training-ready.
- Final rows: **{rows:,}**
- Final usable hours: **{hours:.2f}**
- Manifest: `data/silver_master/train.csv`
- Detailed quality manifest: `data/silver_master/silver_manifest_detailed.csv`
- Gold+Silver curriculum manifests: `data/gold_silver_training/`
- Full report: `reports/silver_quality_report/SILVER_CORPUS_REPORT.md`
- Policy: SILVER is train-only; Gold validation/test remain locked.

| Dataset | Rows | Hours | Known speakers |
|---|---:|---:|---:|
{dataset_lines}"""
    replace_section(DOCS / "DATA_GOVERNANCE.md", section)
    replace_section(
        DOCS / "STATUS.md",
        f"""## SILVER Data Preparation

Completed at `{now}`: **{rows:,} rows / {hours:.2f} hours** survived strict
audio, transcript, Kotib-teacher agreement, and Gold-overlap gates.
The next data milestone is a Gold+Silver curriculum experiment after the LR search
selects its final freeze boundary and learning rates.""",
    )
    replace_section(
        DOCS / "EXPERIMENT_LEDGER.md",
        f"""## SILVER Corpus Build - {now}

Pinned and processed UzbekVoice filtered, IT YouTube, News YouTube, and Tashkent
podcasts. Teacher: pinned `Kotib/uzbek_stt_v1` with forced Uzbek decoding. Final output:
`{rows:,}` rows and `{hours:.2f}` hours. Gold validation/test were excluded from
SILVER and retained unchanged.""",
    )
    state_path = DOCS / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["last_verified_utc"] = now
    state["silver_master"] = {
        "path": "data/silver_master",
        "rows": rows,
        "hours": hours,
        "train_only": True,
        "gold_validation_test_unchanged": True,
        "report": "reports/silver_quality_report/SILVER_CORPUS_REPORT.md",
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
