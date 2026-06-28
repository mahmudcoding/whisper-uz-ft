# Failure Log

This file records failed experiments, bad assumptions, and operational hazards.
Failures are part of project memory and should not be hidden.

## F001 - Raw Whisper Large-v3 Is Poor for Uzbek

Observed result: raw `openai/whisper-large-v3` on Uzbek had WER `1.0522` and CER
`0.4590`.

Cause: weak Uzbek prior and frequent language-prior confusion.

Lesson: forced Uzbek decoding and Uzbek-specific fine-tuning are mandatory.

## F002 - USC Full FT Underperformed Partial FT

Observed result: one-epoch USC full FT with all parameters trainable, BF16, encoder LR
`2e-6`, decoder LR `8e-6` produced WER `0.2221522737`, CER `0.0565825834`.
Protected partial FT baseline produced WER `0.2005258480`, CER `0.0529079419`.

Cause hypothesis: lower encoder acoustic features were over-updated on only about 105h
of USC.

Lesson: full FT is not the default. Prefer decoder and upper/mid encoder adaptation
unless larger data proves otherwise.

## F003 - Early LR Reports Became Stale

Observed result: `reports/lr_search/FINAL_RECOMMENDATION.md` still recommended
encoder 16-31 + decoder with decoder `8e-6` and encoder `5e-6`, but later blockwise
experiments found a better proxy configuration.

Correct current evidence: `phase4x_encoder_bcd_decoder_2e5_bs4_fast` is the best proxy
result: WER `0.1913407821`, CER `0.0484449599`.

Lesson: generated reports are evidence at their timestamp, not always current truth.
Use metrics artifacts and current docs together.

## F004 - High Decoder LR Can Diverge or Degrade

Observed results:

- decoder-only `5e-5` in Phase 1A: WER `2.6401`, CER `2.1442`;
- decoder-only `2e-5` on 30h Phase 2: WER `6.3134`, CER `2.5301`;
- decoder-only `1.2e-5` on 30h degraded after step 400, ending with much worse
  intermediate metrics.

Lesson: high LR is only useful when paired with enough encoder adaptation and should
not be assumed safe for decoder-only training.

## F005 - Full Encoder at 2e-5 Failed Quality

Observed result: `phase4x_full_encoder_decoder_2e5_bs1_safe` reached validation WER
`3.9112`, CER `2.2550` at step 400.

Cause hypothesis: lower encoder layers are too sensitive to aggressive updates.

Lesson: keep encoder 0-7 frozen for the current data scale.

## F006 - Aggressive All-Blocks Batch 2 OOM/Failed

Observed result: `phase4x_main_all_blocks_aggressive_failed_bs2_20260627T054224Z`
failed after about 52 seconds with no usable eval metrics.

Lesson: full/all-block variants need conservative batch sizing and are not the current
promotion path.

## F007 - Silver Teacher Scoring Initially Used the Wrong Teacher Concept

Observed issue: using an in-project fine-tuned model as Silver teacher was rejected as
self-reinforcing and scientifically wrong.

Resolution: use external `Kotib/uzbek_stt_v1` with forced Uzbek decoding.

Lesson: pseudo-label/filter teachers must be independent of the student under search
unless a later self-training stage is explicitly designed.

## F008 - Automatic Language Detection Falsely Rejected Uzbek

Observed issue: the first 3,621 Silver scores from a USC-teacher/language-detection
path were invalidated because accurate Uzbek samples were rejected by automatic
language ID.

Resolution: forced Uzbek decoding with Kotib teacher.

Lesson: do not use automatic language detection as a strict gate for Uzbek speech in
this project.

## F009 - Test Loading During Search Is a Leakage Risk

Observed issue: earlier training pipeline designs could load test splits by default.

Resolution: LR-search configs and active full Gold config use `load_test_split: false`
and `evaluate_test_after_training: false` until final evaluation.

Lesson: validate data access, not only metric names.

## F010 - Zero-Block Freeze Edge Case

Observed issue: generic Python `[-0:]` slicing can select all encoder layers instead
of none.

Resolution: explicit tuning modes and blockwise LR controls in `src/model.py`.

Lesson: always verify actual `requires_grad` state and parameter counts at startup.

## F011 - FeruzaSpeech Gold Inclusion Was Reversed

Observed issue: FeruzaSpeech was initially added to Gold, but later license/trust review
found it should not be in highest-trust Gold.

Resolution: moved to train-only Silver on 2026-06-27; Gold now contains zero Feruza
rows.

Lesson: acquisition success is not enough for Gold inclusion; licensing and trust tier
must be documented.

## F012 - Old Checkpoints Were Deleted During Storage Cleanup

Observed state: many old LR-search checkpoint and final-model files are deleted in git
status. Metrics, reports, logs, and selected models remain.

Lesson: do not assume every historic output directory is resumable. Use metrics for
history and current checkpoints for resume.
