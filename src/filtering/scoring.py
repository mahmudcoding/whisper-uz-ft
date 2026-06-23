from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

from text_normalization import normalize_uzbek_text


SUSPICIOUS_SYMBOL_RE = re.compile(r"""[^0-9A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ\s'`ʻʼ’.,!?;:()/"-]""")


@dataclass
class QualityScore:
    score: float
    decision: str
    reasons: list[str]
    chars_per_second: float | None = None
    transcript_chars: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = "|".join(self.reasons)
        return data


def score_sample(
    transcript: str,
    duration: float,
    asr_similarity: float | None = None,
    asr_cer: float | None = None,
    asr_wer: float | None = None,
) -> QualityScore:
    text = "" if transcript is None else str(transcript).strip()
    normalized_text = normalize_uzbek_text(text)
    reasons: list[str] = []
    score = 100.0
    chars = len(text)
    cps = chars / duration if duration and duration > 0 else None

    if not text:
        score -= 80
        reasons.append("empty_transcript")
    if duration <= 0 or not math.isfinite(duration):
        score -= 80
        reasons.append("invalid_duration")
    if duration > 30:
        score -= 10
        reasons.append("long_utterance")
    if cps is not None:
        if cps < 2:
            score -= 30
            reasons.append("low_chars_per_second")
        if cps > 35:
            score -= 30
            reasons.append("high_chars_per_second")
    if SUSPICIOUS_SYMBOL_RE.search(normalized_text):
        score -= 15
        reasons.append("suspicious_symbols")
    if len(set(text.replace(" ", ""))) <= 2 and chars > 10:
        score -= 25
        reasons.append("low_character_diversity")
    if asr_similarity is not None and asr_similarity < 0.70:
        score -= 35
        reasons.append("low_asr_similarity")
    if asr_cer is not None and asr_cer > 0.35:
        score -= 25
        reasons.append("high_asr_cer")
    if asr_wer is not None and asr_wer > 0.80:
        score -= 20
        reasons.append("high_asr_wer")

    score = max(0.0, min(100.0, score))
    decision = "keep"
    if score < 60:
        decision = "reject"
    elif score < 80 or reasons:
        decision = "suspicious"
    return QualityScore(score=score, decision=decision, reasons=reasons, chars_per_second=cps, transcript_chars=chars)
