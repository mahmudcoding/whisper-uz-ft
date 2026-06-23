from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .scoring import score_sample
except ImportError:
    from scoring import score_sample


def filter_manifest(input_csv: Path, output_csv: Path, bad_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    records = []
    seen_audio = set()
    seen_text = set()
    for idx, row in df.iterrows():
        duration = float(row.get("duration", 0) or 0)
        text = "" if pd.isna(row.get("text")) else str(row.get("text"))
        q = score_sample(text, duration)
        reasons = q.reasons.split("|") if isinstance(q.reasons, str) else list(q.reasons)
        audio_path = str(row.get("audio_path", ""))
        if audio_path in seen_audio:
            q.score = min(q.score, 50)
            q.decision = "reject"
            reasons.append("duplicate_audio_path")
        if text in seen_text:
            q.score = min(q.score, 75)
            if q.decision == "keep":
                q.decision = "suspicious"
            reasons.append("duplicate_transcript")
        seen_audio.add(audio_path)
        seen_text.add(text)
        out = row.to_dict()
        out.update(
            {
                "row_index": idx,
                "quality_score": q.score,
                "quality_decision": q.decision,
                "quality_reasons": "|".join(sorted(set(reasons))),
                "chars_per_second": q.chars_per_second,
                "transcript_chars": q.transcript_chars,
            }
        )
        records.append(out)
    scored = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    scored[scored["quality_decision"] != "keep"].to_csv(bad_csv, index=False)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ASR dataset transcript quality.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bad-csv", required=True)
    args = parser.parse_args()
    scored = filter_manifest(Path(args.input_csv), Path(args.output_csv), Path(args.bad_csv))
    print(scored["quality_decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
