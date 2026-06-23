#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval_suite import evaluate_transformers


TURKISH_WORDS = re.compile(r"\b(ve|değil|için|çok|şimdi|olarak|gibi|kadar)\b", re.IGNORECASE)
TURKISH_CHARS = re.compile(r"[ğşİı]")
KAZAKH_MARKERS = re.compile(r"[әіңүұқөһ]|\b(jane|bolyp|ushin|qazaq)\b", re.IGNORECASE)
RUSSIAN_MARKERS = re.compile(r"[А-Яа-яЁё]")
ENGLISH_MARKERS = re.compile(r"\b(the|and|is|are|with|for|this|that|you|not)\b", re.IGNORECASE)


def classify_confusion(text: str) -> list[str]:
    labels: list[str] = []
    if TURKISH_WORDS.search(text) or TURKISH_CHARS.search(text):
        labels.append("turkish_like")
    if KAZAKH_MARKERS.search(text):
        labels.append("kazakh_like")
    if RUSSIAN_MARKERS.search(text):
        labels.append("russian_cyrillic")
    if ENGLISH_MARKERS.search(text):
        labels.append("english_like")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure non-Uzbek language-prior leakage on Uzbek audio.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", "--dataset", dest="dataset", required=True)
    parser.add_argument("--output", default="reports/language_confusion.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()

    result = evaluate_transformers(
        args.model_path,
        Path(args.dataset),
        args.max_samples,
        "uz",
        max(1, args.batch_size),
        args.precision,
    )
    examples = []
    counts = {"turkish_like": 0, "kazakh_like": 0, "russian_cyrillic": 0, "english_like": 0}
    for item in result.get("examples", []):
        labels = classify_confusion(item["prediction"])
        for label in labels:
            counts[label] += 1
        if labels:
            examples.append({**item, "confusion_labels": labels})
    denom = max(1, len(result.get("examples", [])))
    report = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "samples_scored_for_confusion": denom,
        "confusion_counts": counts,
        "confusion_rates": {key: value / denom for key, value in counts.items()},
        "wer": result.get("wer"),
        "cer": result.get("cer"),
        "normalized_wer": result.get("normalized_wer"),
        "normalized_cer": result.get("normalized_cer"),
        "confusion_examples": examples,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
