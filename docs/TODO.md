# TODO

Generated: 2026-06-23 UTC

## Highest Priority

1. Run normalized evaluation for raw large-v3, final model, Rubai, and Kotib on the same test set.
2. Build a small real-world Uzbek benchmark: meetings, calls, podcasts, webinars, noisy speech.
3. Add teacher-ASR similarity scoring to the filtering pipeline.
4. Clone and pin the Rubai repository commit for exact implementation comparison.
5. Run a 100-300 step BF16 training dry run.

## Uzbek-Only Priority Override

1. Run a 100-300 step full BF16 fine-tuning dry run using `configs/full_ft_uzbek.yaml`.
2. If stable, run 4 epochs full FT and compare with `outputs/final_model`.
3. Run normalized evaluation for raw large-v3, partial FT, full FT, Rubai, and Kotib.
4. Build a real-world Uzbek benchmark before claiming production SOTA.

## Data

1. Review `bad_samples.csv`; do not delete automatically.
2. Add source/license metadata columns for future data.
3. Build long-form and multi-speaker evaluation manifests.
4. Start data collection plan for 500-1500h.

## Training

1. Compare second USC epoch vs adding diverse data.
2. Compare partial fine-tune vs full fine-tune on a controlled subset.
3. Test BF16 optimizer-step stability.
4. Track per-domain WER once real-world data exists.

## Inference

1. Convert final model to CTranslate2/faster-whisper format.
2. Benchmark final model offline and streaming after conversion.
3. Recompute capacity planning with final model numbers.
