# Roadmap

Last cleaned for obsolete documentation sections: `2026-07-01T04:52:10Z`.

## P0

- Let Stage 1 Gold+Silver no-cache run reach step 1000 validation.
- Verify checkpoint save succeeds and disk remains stable.
- Compare validation WER/CER with full-Gold best model.

## P1

- If Stage 1 improves, continue to later evals and preserve best model.
- If Stage 1 underperforms, analyze Silver mix, sampling weights, and teacher-score thresholds.
- Consider Gold-weighted curriculum or lower LR continuation.

## P2

- Expand Silver/Bronze data only after current Stage 1 result is understood.
- Build final test evaluation protocol once a model is selected by validation.
