# Decisions and Rationale

## Decision: Optimize Uzbek Only

Decision:

- Minimize Uzbek WER/CER only.
- Accept catastrophic forgetting of non-Uzbek languages.

Rationale:

- The production target is a dedicated Uzbek ASR model.
- Raw Whisper multilingual priors harm Uzbek output.
- Separate models can serve English, Russian, or multilingual needs.

Expected impact:

- Better Uzbek decoder prior.
- Less Turkish/Kazakh/Russian drift.

Risk:

- Model may become worse for non-Uzbek language recognition.

## Decision: Force Uzbek Decoding

Decision:

- Always set `language="uz"` and `task="transcribe"`.
- Do not allow automatic language detection for Uzbek ASR evaluation or production inference.

Rationale:

- Raw Whisper can choose wrong language priors for Uzbek speech.
- Forced decoding removes one major source of language confusion.

Risk:

- True non-Uzbek speech may be transcribed as Uzbek-like text.

## Decision: Move Beyond Partial FT

Decision:

- Test full FT instead of preserving lower encoder layers frozen.

Rationale:

- Partial FT reached WER `20.05%`, but full Uzbek-only adaptation may better remove harmful multilingual priors.
- The user explicitly prioritizes Uzbek quality over multilingual preservation.

Risk:

- More compute.
- More overfitting risk on only 104.63h USC.
- Requires strict eval and early stopping.

## Decision: Use Layer-Wise LR

Decision:

- Encoder LR `2e-6`.
- Decoder LR `8e-6`.

Rationale:

- Decoder language prior appears to be the main bottleneck.
- Encoder acoustics from Whisper remain valuable.
- Decoder should adapt faster than encoder.

Risk:

- Decoder can overfit or become unstable.
- Monitor WER, hallucination, and sample predictions.

## Decision: Use BF16 on A40

Decision:

- Use BF16 for full FT.

Rationale:

- Local PyTorch reports BF16 support on A40.
- Dry run completed successfully.
- BF16 avoids some FP16 scaling issues.

Risk:

- BF16 behavior can differ by kernel/library; keep dry-run and NaN monitoring.

## Decision: One Epoch for Current Full FT

Decision:

- Current full FT USC run should be one epoch, not four.

Rationale:

- User explicitly changed requirement.
- One epoch provides evidence before committing to multi-day training.

Risk:

- One epoch may underfit relative to 3-4 epochs.
- It is still the right evidence-gathering step.

## Decision: Gold Data Before Silver/Bronze

Decision:

- Build Gold master corpus first.
- Do not start Silver until Gold execution is complete.

Rationale:

- Clean, normalized, deduplicated Gold data is the foundation for trustworthy evaluation and teacher training.
- Noisy data can hurt if introduced too early.

Risk:

- Progress toward 500-1500h is delayed until Gold is stable.

## Decision: Archive Instead of Delete

Decision:

- Historical docs and baseline artifacts are moved to archive, not deleted.

Rationale:

- Prevents losing experiment history.
- Keeps current docs minimal while preserving forensic traceability.

Risk:

- Future agents may read archived docs and mistake them for current state. The numbered docs are authoritative.
