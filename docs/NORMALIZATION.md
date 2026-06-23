# Uzbek Text Normalization

Generated: 2026-06-23 UTC

## Module

Implemented:

- `src/text_normalization/__init__.py`
- `src/text_normalization/uz_normalizer.py`
- `src/text_normalization/tests.py`

Validation:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
PYTHONPATH=src python -m text_normalization.tests
```

Latest result:

- Tests passed: 5
- Speed: about 2.06M chars/sec

## Features

The normalizer handles:

- Uzbek Latin lowercasing and whitespace cleanup.
- Uzbek Cyrillic to Latin conversion.
- Mixed-script Uzbek normalization.
- Apostrophe variants: `'`, `’`, `` ` ``, `ʻ`, `ʼ`, and related Unicode marks.
- Uzbek letters: `ў -> o'`, `қ -> q`, `ғ -> g'`, `ҳ -> h`.
- Punctuation normalization for quotes, dashes, ellipses, and common non-ASCII punctuation.
- Optional punctuation removal.

## Usage

```python
from text_normalization import normalize_uzbek_text

text = normalize_uzbek_text("Ғўза ўсимлиги O‘ZBEKCHA")
assert text == "g'oza o'simligi o'zbekcha"
```

## Design Notes

The normalizer intentionally preserves Russian and English letters so code-switching is not destroyed. This matters for enterprise Uzbek meetings where Russian and English terms appear inside Uzbek speech.

Number normalization is exposed as a configuration field but not yet expanded into Uzbek word forms. That should be added only with a tested numeral grammar because partial number conversion can increase WER.

