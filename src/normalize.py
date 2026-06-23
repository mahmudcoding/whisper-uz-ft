from __future__ import annotations

import re
import unicodedata


SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_uzbek_text(text: object, lowercase: bool = False) -> str:
    """Normalize Uzbek Latin transcripts while preserving Uzbek apostrophe letters."""
    if text is None:
        return ""
    value = str(text)
    value = unicodedata.normalize("NFC", value)
    value = CONTROL_RE.sub(" ", value)
    value = value.replace("ʼ", "‘").replace("`", "‘")
    value = re.sub(r"\bo'", "o‘", value)
    value = re.sub(r"\bg'", "g‘", value)
    value = re.sub(r"\bO'", "O‘", value)
    value = re.sub(r"\bG'", "G‘", value)
    value = value.replace("'", "‘")
    if lowercase:
        value = value.lower()
    value = SPACE_RE.sub(" ", value).strip()
    return value


def is_valid_transcript(text: object, min_chars: int = 2, max_chars: int = 500) -> bool:
    value = normalize_uzbek_text(text)
    return min_chars <= len(value) <= max_chars
