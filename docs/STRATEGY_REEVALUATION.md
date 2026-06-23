# Strategy Reevaluation: Uzbek-Only ASR

Generated: 2026-06-23 UTC

## Decision

For the stated objective, full fine-tuning is now the preferred next experiment.

Previous partial fine-tuning was a conservative multilingual-preservation strategy. That assumption is invalid for this mission. If English/Russian/Turkish preservation does not matter, freezing the lower 24 encoder blocks may preserve exactly the multilingual acoustic/language priors that caused raw Whisper large-v3 to emit Turkish/Kazakh-like text on Uzbek speech.

## Evidence

Current results:

| Model | Strategy | WER | CER |
| --- | --- | ---: | ---: |
| Raw Whisper large-v3 | no FT | 1.0522 | 0.4590 |
| Mini FT | partial | 0.4961 | 0.1094 |
| Full USC run | partial, 1 epoch | 0.2005 | 0.0529 |

Partial fine-tuning worked, but it does not prove it is optimal. It proves that Uzbek adaptation is valuable.

## Option A: Partial Fine-Tuning

Rationale:

- Lower risk of overfitting on 104h USC.
- Lower optimizer-state memory.
- Already validated on A40.

Expected impact:

- Good USC performance.
- Lower chance of catastrophic model drift.

Risk:

- Frozen encoder layers retain multilingual priors.
- May cap Uzbek performance.
- Less able to adapt to Uzbek phonotactics, apostrophes, and script conventions.

## Option B: Full Fine-Tuning

Rationale:

- Objective is Uzbek-only quality.
- Catastrophic forgetting is acceptable.
- Full FT lets every acoustic and decoder representation specialize for Uzbek.
- RubaiSTT v2 reportedly uses full fine-tuning.

Expected impact:

- Best chance of reducing WER/CER below the current partial-FT plateau.
- Better chance of eliminating Turkish/Kazakh prior leakage.

Risk:

- Higher VRAM due optimizer states.
- Higher overfitting risk on clean read speech.
- More likely to degrade out-of-domain robustness if trained only on USC.

## Recommendation

Run full fine-tuning with BF16, low LR, 4 epochs, and strict validation/test tracking. Stop based on normalized Uzbek validation WER, not multilingual retention.

Expected impact:

- Potential USC test WER improvement from `0.2005` toward `0.15-0.18`.
- Larger real-world gains require data scaling, not just more USC epochs.

Risk:

- If trained only on USC, full FT may overfit read speech and underperform on meetings/calls. Mitigate by adding a real-world eval set before declaring SOTA.

