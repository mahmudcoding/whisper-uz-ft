from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


APOSTROPHES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u02bb": "'",
    "\u02bc": "'",
    "\u02bd": "'",
    "\u02be": "'",
    "\u02bf": "'",
    "\u0060": "'",
    "\u00b4": "'",
    "\u2032": "'",
    "\u201a": "'",
    "\uff07": "'",
    "\u02ca": "'",
}


CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ц": "s",
    "ч": "ch",
    "ш": "sh",
    "ъ": "'",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "ў": "o'",
    "қ": "q",
    "ғ": "g'",
    "ҳ": "h",
}


PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "؛": ";",
        "؟": "?",
        "–": "-",
        "—": "-",
        "…": "...",
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
    }
)


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    cyrillic_to_latin: bool = True
    normalize_apostrophes: bool = True
    normalize_punctuation: bool = True
    remove_punctuation: bool = False
    normalize_numbers: bool = False
    keep_russian_english: bool = True


class UzbekNormalizer:
    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def normalize(self, text: str | None) -> str:
        if text is None:
            return ""
        text = unicodedata.normalize("NFKC", str(text))
        if self.config.normalize_apostrophes:
            text = "".join(APOSTROPHES.get(ch, ch) for ch in text)
        if self.config.normalize_punctuation:
            text = text.translate(PUNCT_TRANSLATION)
        if self.config.cyrillic_to_latin:
            text = self._cyrillic_to_latin(text)
        if self.config.lowercase:
            text = text.lower()
        text = self._normalize_uzbek_digraphs(text)
        if self.config.remove_punctuation:
            text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
        else:
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
            text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
        text = re.sub(r"'+", "'", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _cyrillic_to_latin(self, text: str) -> str:
        chars: list[str] = []
        for ch in text:
            low = ch.lower()
            repl = CYRILLIC_TO_LATIN.get(low)
            if repl is None:
                chars.append(ch)
            elif ch.isupper() and repl:
                chars.append(repl.capitalize())
            else:
                chars.append(repl)
        return "".join(chars)

    @staticmethod
    def _normalize_uzbek_digraphs(text: str) -> str:
        text = re.sub(r"\bo\s*['`ʻʼ’]\s*", "o'", text, flags=re.IGNORECASE)
        text = re.sub(r"\bg\s*['`ʻʼ’]\s*", "g'", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsh\b", "sh", text, flags=re.IGNORECASE)
        text = re.sub(r"\bch\b", "ch", text, flags=re.IGNORECASE)
        return text


def normalize_uzbek_text(text: str | None, config: NormalizationConfig | None = None) -> str:
    return UzbekNormalizer(config).normalize(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()
    print(normalize_uzbek_text(" ".join(args.text)))
