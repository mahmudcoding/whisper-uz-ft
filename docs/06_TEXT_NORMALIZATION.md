# Text Normalization

## Purpose

Uzbek ASR evaluation and training are highly sensitive to script and apostrophe inconsistency. The project standard is canonical Uzbek Latin.

One spoken phrase should map to one canonical written form wherever possible.

Implementation:

- `src/text_normalization/uz_normalizer.py`
- `src/text_normalization/tests.py`

Main API:

```python
from text_normalization import normalize_uzbek_text
text = normalize_uzbek_text(raw_text)
```

## Default Behavior

Default `NormalizationConfig`:

- `lowercase=True`
- `cyrillic_to_latin=True`
- `normalize_apostrophes=True`
- `normalize_punctuation=True`
- `remove_punctuation=False`
- `normalize_numbers=False`
- `keep_russian_english=True`

## Apostrophe Normalization

All common apostrophe variants map to ASCII `'`.

Handled variants include:

- `‘`
- `’`
- `‛`
- `ʻ`
- `ʼ`
- `ʽ`
- `ʾ`
- `ʿ`
- backtick
- acute accent
- prime
- fullwidth apostrophe

Canonical Uzbek forms:

- `o‘`, `oʻ`, `oʼ`, `o’` -> `o'`
- `g‘`, `gʻ`, `gʼ`, `g’` -> `g'`

## Cyrillic to Latin Mapping

Important Uzbek-specific mappings:

| Cyrillic | Latin |
|---|---|
| `ў` | `o'` |
| `қ` | `q` |
| `ғ` | `g'` |
| `ҳ` | `h` |
| `ч` | `ch` |
| `ш` | `sh` |
| `ё` | `yo` |
| `ю` | `yu` |
| `я` | `ya` |
| `ъ` | `'` |
| `ь` | empty |

The implementation also maps the core Cyrillic alphabet to Latin approximations.

## Punctuation and Unicode

The normalizer applies Unicode NFKC normalization and maps punctuation variants:

- Chinese comma/period variants to ASCII forms.
- Arabic semicolon/question mark to ASCII forms.
- En dash and em dash to `-`.
- Ellipsis to `...`.
- Curly quotes to straight quotes.

Whitespace is collapsed to single spaces.

## Punctuation Removal Mode

When `remove_punctuation=True`, punctuation other than word characters, whitespace, and apostrophe is removed.

Use this for normalized WER/CER evaluation if strict punctuation should not count against ASR.

## Known Limitations

- Number normalization is currently optional but not fully expanded into spoken Uzbek number words.
- Russian and English code-switch text is preserved where possible.
- Some Cyrillic mappings are approximate; high-quality mixed-script code-switch evaluation needs curated test cases.

## Tests

Run:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src
python src/text_normalization/tests.py
```

The tests should cover apostrophes, Cyrillic conversion, punctuation cleanup, whitespace cleanup, and speed.
