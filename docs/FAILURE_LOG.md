# Failure Log and Lessons

**Document role:** Prevent recurrence of known failures and incorrect assumptions.

## F001 - Raw Whisper Hallucination and Language Drift

**Symptom:** WER above 100%, repeated tokens, Turkish/Kazakh-like text.

**Cause:** Weak Uzbek multilingual prior and unconstrained generation behavior.

**Resolution:** Force Uzbek decoding and fine-tune.

**Prevention:** Never deploy raw large-v3 for Uzbek.

## F002 - Smoke Benchmarks Misrepresented Capacity

**Symptom:** Early capacity estimates were based on seconds of audio.

**Cause:** Startup overhead and workload shape dominated results.

**Resolution:** Build a 5h long-form benchmark.

**Prevention:** Use smoke only for functional validation; use long-duration measured
workloads for planning.

## F003 - Excessive Evaluation Overhead

**Symptom:** Full-FT runtime estimates included disproportionate validation time.

**Cause:** Evaluation/checkpoint cadence was too frequent for autoregressive generation.

**Resolution:** Use larger intervals for long training; LR search uses explicit proxy
cadence because evaluation is part of candidate screening.

**Prevention:** Estimate generation cost before setting `eval_steps`.

## F004 - Early-Stopping Metric Mismatch

**Symptom:** Early stopping expected `eval_wer` but saw `test_wer`.

**Cause:** Validation and test metric prefixes were conflated.

**Resolution:** Validation emits `eval_*`; final test emits `test_*`;
`metric_for_best_model: wer`, `greater_is_better: false`.

**Prevention:** Test callbacks against a real validation event.

## F005 - Four-Epoch Job Started Before Final Requirement

**Symptom:** A multi-day full-FT run launched with four epochs; user required one.

**Cause:** Launch occurred before final epoch decision was stabilized.

**Resolution:** Patch to one epoch, wait for checkpoint, stop, and resume.

**Prevention:** Before any long launch, print and verify resolved epoch count, LRs,
freeze mode, precision, output path, and resume behavior.

## F006 - Resume Blocked by PyTorch 2.5.1

**Symptom:** Transformers refused optimizer/scheduler `torch.load`.

**Cause:** Installed PyTorch was below the security-required version.

**Resolution:** Upgrade to torch 2.7.1+cu126, torchvision 0.22.1+cu126, torchaudio
2.7.1+cu126; validate CUDA/BF16 and `pip check`; resume successfully.

**Prevention:** Treat dependency versions as part of checkpoint compatibility.

## F007 - Full FT Degraded WER/CER

**Symptom:** Full FT WER 22.22% versus partial FT 20.05%.

**Cause hypothesis:** Lower encoder acoustic features were over-updated on 104.63h of
clean read speech.

**Resolution:** Decoder-first LR search and conservative upper-encoder unfreezing.

**Prevention:** Do not equate more trainable parameters with better quality.

## F008 - FeruzaSpeech Access Blocked

**Symptom:** Dataset could not be downloaded.

**Cause:** `k2speech/FeruzaSpeech` requires manual gated access; no token configured.

**Resolution:** Document the blocker and proceed without silently claiming Feruza hours.

**Prevention:** Verify access/licensing before corpus planning.

## F009 - TorchCodec Audio Decode Failure

**Symptom:** Hugging Face audio decode failed during FLEURS export.

**Cause:** Optional TorchCodec path was unavailable/incompatible.

**Resolution:** Use `Audio(decode=False)` and explicit SoundFile decoding.

**Prevention:** Prefer explicit, testable audio decode paths.

## F010 - Documentation Sprawl and Contradiction

**Symptom:** Eighteen top-level docs duplicated state, plans, history, and commands;
several statements were stale.

**Cause:** Topic docs accumulated without clear ownership boundaries.

**Resolution:** On 2026-06-24, archive the complete old set and replace it with
role-based manuals, one live status page, separate ledgers, machine-readable state, and
a stricter validator.

**Prevention:** Follow `DOCUMENTATION_STANDARD.md`; do not create a new doc when an
existing owner document covers the topic.

## F011 - Dry-Run Runner Created Output Directories

**Symptom:** Config validation polluted `outputs_lr_search/`, causing real runs to use
timestamped IDs.

**Cause:** Dry-run wrote metadata before returning.

**Resolution:** Make dry-run side-effect free and archive the temporary directories.

**Prevention:** Validation modes must not mutate experiment state.

## F012 - Test Was Preprocessed During Search

**Symptom:** Search disabled final test evaluation but still loaded/preprocessed
`test.csv`.

**Cause:** Dataset construction always included all splits.

**Resolution:** Add `load_test_split: false`; enforce in configs and runner; hash test
manifests; reject search metrics containing test results.

**Prevention:** Audit data access, not only metric emission.

## F013 - Freeze Mode Zero-Block Edge Case

**Symptom:** Python `[-0:]` would select all encoder blocks.

**Cause:** Generic "last N blocks" slicing did not handle zero explicitly.

**Resolution:** Add explicit tuning modes and validate zero before slicing.

**Prevention:** Verify actual `requires_grad` state and parameter counts on the real
model.
