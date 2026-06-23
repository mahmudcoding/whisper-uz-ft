# Failures and Lessons

## Raw Whisper Is Not Good Enough for Uzbek

Failure:

- Raw `openai/whisper-large-v3` produced very high WER/CER and hallucinations.
- Examples include repeated text and Turkish/Kazakh-like outputs.

Lesson:

- Do not deploy raw Whisper large-v3 for Uzbek.
- Force Uzbek decoding and fine-tune.

## Smoke Inference Benchmarks Were Insufficient

Failure:

- Early inference benchmarking on smoke datasets was too small for enterprise capacity planning.

Lesson:

- Use long-form offline throughput benchmarks for production planning.
- Keep offline and streaming benchmarks separate.

## Evaluation Overhead Was Too High

Failure:

- Earlier full FT estimates had excessive eval/checkpoint overhead.

Lesson:

- Use `eval_steps: 1000` and `save_steps: 1000` for full FT unless debugging.
- Avoid frequent full validation during multi-day runs.

## Early Stopping Metric Mismatch

Failure:

- Early stopping expected `eval_wer` while some evaluation paths emitted `test_wer`.

Lesson:

- Validation must emit `eval_wer`.
- Final test must emit `test_wer`.
- `metric_for_best_model: wer` and `greater_is_better: false` are required.

## Four-Epoch Run Was Launched Before Final User Decision

Failure:

- Full FT was launched with `epochs: 4`; user later required only one epoch.

Mitigation:

- Patched config to `epochs: 1`.
- Added `scripts/guard_one_epoch_resume.sh` to restart from checkpoint 1000 with the patched config.
- Guard completed checkpoint wait and restart, but resume failed due PyTorch/Transformers compatibility.

Lesson:

- Before long training, always restate final epoch count and confirm config.
- Training launch docs must capture current user decision.

## Resume Failed With PyTorch 2.5.1

Failure:

- Restart from `outputs_full_ft/checkpoint-1000` failed while loading optimizer/scheduler state.
- Transformers `5.12.1` rejects `torch.load` for `.pt` state files unless PyTorch is at least `2.6`.
- Local PyTorch is `2.5.1+cu121`.

Lesson:

- Resume support must be tested with real optimizer/scheduler checkpoint loading, not only checkpoint discovery.
- Dependency versions are part of the training contract.
- Preferred fix is upgrading PyTorch to `>=2.6` before resuming.

## FeruzaSpeech Acquisition Blocked

Failure:

- `k2speech/FeruzaSpeech` is gated manual on Hugging Face and no HF token is configured.

Lesson:

- Document exact dataset blockers.
- Do not silently skip gated datasets.

## TorchCodec Issue During FLEURS Export

Failure:

- Hugging Face audio decoding path required TorchCodec and failed in this environment.

Mitigation:

- Export script uses `Audio(decode=False)` and decodes bytes/path with `soundfile`.

Lesson:

- Prefer robust explicit audio decoding over relying on optional HF decoder backends.

## Documentation Sprawl

Failure:

- Many overlapping docs existed with stale or contradictory details.

Mitigation:

- Archived legacy docs under `docs/archive/legacy_20260623T113818Z/`.
- Rebuilt required numbered docs as single source of truth.

Lesson:

- Future agents must update numbered docs after meaningful changes.
