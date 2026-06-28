from __future__ import annotations

from difflib import SequenceMatcher

from jiwer import cer, wer

try:
    from text_normalization import NormalizationConfig, normalize_uzbek_text
except Exception:  # pragma: no cover
    from src.text_normalization import NormalizationConfig, normalize_uzbek_text


AGREEMENT_NORMALIZATION = NormalizationConfig(remove_punctuation=True)


def normalize_for_agreement(text: str) -> str:
    """Canonicalize ASR text without scoring punctuation differences."""
    return normalize_uzbek_text(text, AGREEMENT_NORMALIZATION)


def normalized_similarity(reference: str, hypothesis: str) -> float:
    ref = normalize_for_agreement(reference)
    hyp = normalize_for_agreement(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    return float(SequenceMatcher(None, ref, hyp).ratio())


def normalized_wer(reference: str, hypothesis: str) -> float:
    return float(wer(normalize_for_agreement(reference), normalize_for_agreement(hypothesis)))


def normalized_cer(reference: str, hypothesis: str) -> float:
    return float(cer(normalize_for_agreement(reference), normalize_for_agreement(hypothesis)))
