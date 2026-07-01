# Failure Log and Lessons

Last rebuilt: `2026-07-01T04:50:03Z`.

## Stage 1 Cached Run Disk-Full Failure

- Failed run output: `outputs_stage1_gold_silver/`.
- Failure point: step 1000 after successful validation and best-model snapshot.
- Metrics before failure: validation WER 29.42%, CER 7.00%.
- Error: `safetensors... I/O error: No space left on device (os error 28)` while writing `checkpoint-1000`.
- Root cause: large Hugging Face dataset/feature cache plus checkpoint growth filled disk.
- Consequence: no valid resumable checkpoint existed; only a standalone best model snapshot existed. The failed run was deleted by user request.
- Fix applied: `src/train.py` now disables HF datasets caching and uses lazy `OnTheFlySpeechDataset`; Stage 1 writes to `outputs_stage1_gold_silver_nocache/`; `save_total_limit: 2`.

## Full FT USC Degraded Quality

- Partial FT USC baseline: WER 20.05%, CER 5.29%.
- Full FT USC: WER 22.22%, CER 5.66%.
- Lesson: updating all layers on ~104h can damage useful acoustic features. Lower encoder A should stay frozen unless strong evidence supports otherwise.

## LR-Search Failures

- Decoder-only high LR `5e-5` diverged badly in Phase 1A: WER 264.01%, CER 214.42% at step 150.
- Full encoder plus decoder at 2e-5 collapsed on proxy: WER 391.12%, CER 225.50% at step 400.
- `phase4x_main_all_blocks_aggressive` with A tiny/B/C/D/decoder schedule was weak at step 200 and the checkpoint was later deleted during disk cleanup.
- Batch-size speed experiments showed higher memory utilization is possible, but exact fairness against prior LR runs was intentionally abandoned when speed became more important.

## Documentation Drift

- Several docs previously became stale because experiments, dataset migrations, and cleanup happened faster than documentation updates.
- Current policy: docs are rebuilt from actual artifacts and must be updated after meaningful code/config/data/training changes.
