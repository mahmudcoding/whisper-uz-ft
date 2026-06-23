from __future__ import annotations

import time

from .uz_normalizer import normalize_uzbek_text


CASES = {
    "Ўзбекистон — катта давлат.": "o'zbekiston - katta davlat.",
    "Ғишт, Қўқон, Ҳамза": "g'isht, qo'qon, hamza",
    "Oʻzbekiston va g‘alaba": "o'zbekiston va g'alaba",
    "  Salom   dunyo !  ": "salom dunyo!",
    "Ш, Ч, шahar": "sh, ch, shahar",
}


def run_tests() -> None:
    for src, expected in CASES.items():
        actual = normalize_uzbek_text(src)
        assert actual == expected, (src, actual, expected)

    sample = "Oʻzbekiston Республикаси. Ғалаба!" * 1000
    t0 = time.perf_counter()
    for _ in range(1000):
        normalize_uzbek_text(sample)
    elapsed = time.perf_counter() - t0
    print({"tests": len(CASES), "benchmark_chars_per_sec": int(len(sample) * 1000 / elapsed)})


if __name__ == "__main__":
    run_tests()
