# Decision Log

This file records decisions that future agents must not silently undo. Each decision
lists evidence, impact, risk, and reversal condition.

## D001 - Optimize Uzbek Only

Decision: optimize Uzbek WER/CER only. Catastrophic forgetting of English, Russian,
Turkish, or other Whisper languages is acceptable.

Evidence: raw Whisper large-v3 produced severe Uzbek errors and Turkish/Kazakh-like
outputs; the product goal is Uzbek ASR, not multilingual ASR.

Impact: enables forced Uzbek decoding, Uzbek-only normalization, aggressive decoder
adaptation, and rejection of multilingual-preservation constraints.

Risk: the resulting model may be poor for non-Uzbek speech.

Reversal condition: product requirements explicitly require multilingual transcription
from the same model.

## D002 - Force Uzbek Decoding

Decision: every training/evaluation/inference path must set `language="uz"` and
`task="transcribe"`; automatic language detection is not trusted for Uzbek experiments.

Evidence: language-prior errors were a major baseline failure mode. `src/model.py`
sets forced decoder IDs with `processor.get_decoder_prompt_ids(language="uz",
task="transcribe")`.

Impact: reduces language confusion and stabilizes Uzbek output.

Risk: non-Uzbek audio will be forced into Uzbek-like text.

Reversal condition: a separate language-routing system is introduced and evaluated.

## D003 - Protect the USC Partial-FT Baseline

Decision: `models/partial_ft_usc_baseline/` is immutable.

Evidence: it is the best completed locked-test model: WER `0.2005258480`, CER
`0.0529079419`.

Impact: provides a stable fallback and comparison point.

Risk: storage cost.

Reversal condition: never delete this artifact solely because a better model exists.
Archive newer winners separately.

## D004 - Do Not Assume Full FT Is Better

Decision: full fine-tuning all layers is rejected as the default strategy for current
small or mid-size corpora.

Evidence: one-epoch USC full FT with all 1.543B parameters trainable, BF16, encoder LR
`2e-6`, decoder LR `8e-6` produced WER `0.2221522737`, CER `0.0565825834`, worse than
the protected partial FT baseline.

Impact: future searches should adapt decoder and upper/mid encoder first; lower encoder
acoustic features are likely valuable.

Risk: freezing lower layers may cap performance on much larger or very different data.

Reversal condition: larger diverse data shows validated gains from deeper/full encoder
updates without overfitting.

## D005 - Use Validation-Only Search

Decision: LR search and model selection must not load or evaluate test data.

Evidence: `reports/lr_search/data_leakage_audit.json` passes; configs disable
`load_test_split` and `evaluate_test_after_training`.

Impact: keeps test data valid for final evaluation.

Risk: fewer measurements during search.

Reversal condition: none for hyperparameter/model selection.

## D006 - Use Canonical Uzbek Latin Normalization

Decision: normalize all training and evaluation text to canonical Uzbek Latin.

Evidence: mixed Latin/Cyrillic, apostrophe variants, punctuation, and Unicode variants
inflate WER/CER and fragment targets.

Impact: lower metric noise and more consistent decoder supervision.

Risk: script-preserving output is not supported by default.

Reversal condition: product requires script-preserving ASR and a separate evaluation
policy is created.

## D007 - Treat FeruzaSpeech as Silver, Not Gold

Decision: FeruzaSpeech is excluded from Gold and included only as train-only Silver.

Evidence: 2026-06-27 migration report moved 12,854 rows / 57.8279h out of Gold because
the gated K2Speech terms are restrictive and lower trust than Gold governance.

Impact: Gold remains high-trust and easier to reason about; Feruza can still help
training through down-weighted Silver curriculum.

Risk: less Gold training data.

Reversal condition: licensing/trust review explicitly upgrades FeruzaSpeech.

## D008 - Use Kotib, Not Our Own Model, for Silver Teacher Scoring

Decision: Silver teacher scoring must use `Kotib/uzbek_stt_v1` rather than a current
in-project fine-tuned model.

Evidence: using our own partially trained model as teacher would reinforce its errors.
Kotib was validated with forced Uzbek decoding and exact normalized agreement on the
small validation sample in `reports/silver_quality_report/KOTIB_TEACHER_VALIDATION.md`.

Impact: reduces self-training bias during Silver filtering.

Risk: Kotib itself has biases and is Whisper-medium based.

Reversal condition: an independently stronger teacher is validated.

## D009 - Use Blockwise LR for Current Gold Promotion

Decision: the active full Gold run uses blockwise LR with encoder 0-7 frozen and
encoder 8-31 plus decoder trained at `2e-5`.

Evidence: 30h proxy experiment `phase4x_encoder_bcd_decoder_2e5_bs4_fast` achieved the
best validation WER `0.1913407821` and CER `0.0484449599`, beating decoder-only,
upper-encoder-only, encoder 16-31, full-encoder, and more aggressive alternatives.

Impact: current full Gold run is a direct promotion test of the best proxy schedule.

Risk: proxy validation may not fully predict 207h training; the 30h validation set is
only about 1h.

Reversal condition: full Gold validation or final locked test fails to beat the protected
baseline or shows instability.

## D010 - Use Faster Training Settings Only When Measured Safe

Decision: batch 4, gradient accumulation 8, BF16, gradient checkpointing disabled, and
duration bucketing are allowed for the active B/C/D + decoder run.

Evidence: proxy run `phase4x_encoder_bcd_decoder_2e5_bs4_fast` completed with about
38.1 GiB peak VRAM and stable metrics. The full Gold train max duration is not worse
than the proxy train max duration.

Impact: about 4.5-4.7 sec/optimizer step on A40 for the active run.

Risk: validation has some longer clips and may use more memory; monitor eval at step
1000.

Reversal condition: OOM or instability requires returning to batch 2/1 or enabling
gradient checkpointing.

## D011 - Keep Gold Validation/Test Clean When Adding Silver

Decision: Silver is train-only unless a new explicit evaluation design is approved.
Gold validation and test membership must not change when adding Silver.

Evidence: `data/gold_silver_training/` keeps Gold validation/test unchanged and adds
FeruzaSpeech only to train.

Impact: preserves comparability.

Risk: model selection remains Gold-domain biased.

Reversal condition: a separate Silver/domain validation set is constructed without
touching locked Gold test.
