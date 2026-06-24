# LR Search Implementation Summary

## Status

The learning-rate search framework is implemented and validated. No training experiment
was launched during framework construction.

## Dataset Proxies

| Proxy | Train hours | Samples | USC | Common Voice | FLEURS | Validation |
|---|---:|---:|---:|---:|---:|---:|
| coarse_10h | 9.9971 | 8,733 | 50.01% | 40.00% | 9.99% | 0.9988h |
| main_30h | 29.9987 | 26,249 | 50.00% | 40.00% | 10.00% | 1.0002h |

Validation passed for duration targets, source proportions, missing paths, audio-path
overlap, and reliable speaker IDs. FLEURS speaker isolation cannot be guaranteed because
its exported manifest lacks reliable speaker identity.

## Files Created

- Five scripts under `scripts/lr_search/` for subset creation, validation, freeze
  verification, experiment execution, comparison, and plots.
- Two shared configs and eleven experiment configs under `configs/lr_search/`.
- Two manifest sets under `data/lr_search/`.
- Subset, validation, and freeze reports under `reports/lr_search/`.
- `docs/TRAINING_AND_SEARCH.md`.

## Files Modified

- `src/model.py`: explicit tuning modes, safe zero-block behavior, group diagnostics.
- `src/train.py`: inherited configs, tuning-mode support, deterministic seeds, detailed
  parameter reports, optional test evaluation, and structured run metrics.
- `scripts/update_docs.py`: recognizes the required LR-search plan.
- Authoritative project documentation was updated for the new search phase.

Backups of `src/model.py` and `src/train.py` were created before modification.

## Verified Freeze Modes

| Mode | Trainable params | Trainable share | Expected state |
|---|---:|---:|---|
| decoder_only | 906,521,600 | 58.73% | All encoder layers frozen |
| encoder_24_31_plus_decoder | 1,063,930,880 | 68.93% | Encoder 24-31 + decoder |
| encoder_16_31_plus_decoder | 1,221,340,160 | 79.13% | Encoder 16-31 + decoder |

The verifier loaded real `openai/whisper-large-v3` weights and passed all state checks.

## Experiment Outputs

Each run writes a resolved config, experiment metadata, train log, GPU telemetry,
checkpoints, structured metrics, and train/validation loss plots. The comparison tool
writes CSV and Markdown rankings plus WER, CER, and LR-vs-WER plots.

## Risks and Assumptions

1. Proxy ranking may not perfectly predict 207h training. Confirm on 30h before promotion.
2. FLEURS speaker IDs are unreliable; source split and path separation are still enforced.
3. A 1h validation proxy makes close WER values statistical ties. Use a 0.5 absolute
   WER-point practical tie threshold.
4. `8e-6` is a placeholder decoder LR in Phase 2/3 configs. Override it if Phase 1
   selects another value.
5. Decoder-only still trains about 906.5M unique parameters.
6. Search runs intentionally do not evaluate test data. Test is a one-time promotion step.

## Exact Run Order

1. Run decoder LRs `2e-6`, `8e-6`, `2e-5`, `5e-5`.
2. Compare results.
3. Confirm the winner with `decoder_best_main.yaml`.
4. Run upper encoder LRs `5e-7`, `1e-6`, `2e-6`, `5e-6`.
5. Compare decoder-only against upper-encoder candidates.
6. Run freeze boundaries 23 and 15 using the winning LRs.
7. Lock the best validation configuration and evaluate the test proxy once.
8. Promote only if the result justifies a full Gold-corpus experiment.
