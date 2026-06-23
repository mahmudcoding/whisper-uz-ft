from __future__ import annotations

from difflib import SequenceMatcher

from jiwer import cer, wer

try:
    from text_normalization import normalize_uzbek_text
except Exception:  # pragma: no cover
    from src.text_normalization import normalize_uzbek_text


def normalized_similarity(reference: str, hypothesis: str) -> float:
    ref = normalize_uzbek_text(reference, None)
    hyp = normalize_uzbek_text(hypothesis, None)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    return float(SequenceMatcher(None, ref, hyp).ratio())


def normalized_wer(reference: str, hypothesis: str) -> float:
    return float(wer(normalize_uzbek_text(reference), normalize_uzbek_text(hypothesis)))


def normalized_cer(reference: str, hypothesis: str) -> float:
    return float(cer(normalize_uzbek_text(reference), normalize_uzbek_text(hypothesis)))
