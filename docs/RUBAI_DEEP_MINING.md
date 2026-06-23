# RubaiSTT v2 Deep Mining Report

Generated: 2026-06-23 UTC

Reference repository: `/home/mahmud/rubaistt_v2-open-sourced`

## Scope

I read the full non-git repository contents, including training code, dataset construction, filtering, normalization utilities, inference/demo code, bot code, notebooks, README files, and command snippets. The goal was not to reproduce RubaiSTT, but to extract ideas that improve an Uzbek-only Whisper large-v3 pipeline.

## High Impact, Must Adopt

### 1. Data diversity is their main advantage

Rubai combines podcast speech, news, IT YouTube speech, Common Voice, FLEURS, and USC. This is more important than their training code. Their dataset is about 500h and covers more conversational, public-media, and dialect variation than USC alone.

- Rationale: USC is clean read speech. Enterprise ASR needs meetings, podcasts, calls, webinars, and noisy spontaneous speech.
- Expected impact: largest WER/CER improvement outside the current USC distribution.
- Risk: weak pseudo-labeling or filtering can add label noise; every new source needs normalization and quality scoring.

### 2. Pseudo-labeling with context

Rubai uses Gemini-based transcription for unlabeled/weakly labeled audio and provides previous-segment context. The prompt forces Uzbek Latin, preserves Russian/English words, keeps conversational grammar, and returns empty output for music or sung content.

- Rationale: contextual pseudo-labeling reduces segment-boundary errors and improves natural conversational text.
- Expected impact: high for podcasts/interviews/meetings.
- Risk: LLM hallucination, over-normalization, and licensing/cost constraints. Use teacher agreement filtering before training.

### 3. Filtering with teacher ASR plus fuzzy similarity

Rubai evaluates corpus pairs with a teacher ASR model and computes WER/CER plus `rapidfuzz.fuzz.ratio`. Their reject rule is conservative: reject only when both WER is worse than `0.4` and similarity is below `0.8`.

- Rationale: similarity catches script/format variants that WER over-penalizes; WER catches semantic mismatch.
- Expected impact: high for pseudo-labeled or scraped data.
- Risk: their OR keep rule may retain noisy samples. For our large-v3 pipeline, start conservative and tune thresholds by manually reviewing score bands.

### 4. Strong Uzbek normalization

Rubai's `w_utils/new_text_normalizer.py` handles Cyrillic-to-Latin conversion, apostrophe normalization, Uzbek letters, punctuation cleanup, number-to-words, years, percentages, times, ranges, floats, formatted numbers, and leading-zero expressions.

- Rationale: raw Whisper had severe script/language confusion; canonical labels and normalized metrics are mandatory.
- Expected impact: high for both training stability and fair evaluation.
- Risk: Rubai removes characters outside a narrow Latin punctuation set, which can harm Russian/English code-switching and names. Adopt the ideas, not the exact destructive regex.

### 5. Full fine-tuning with BF16 and SpecAugment

Rubai fully fine-tunes Whisper Medium with BF16, 4 epochs, LR `8e-6`, warmup ratio `0.1`, weight decay `0.03`, max grad norm `3.0`, gradient checkpointing, and SpecAugment (`mask_time_prob=0.03`, `mask_feature_prob=0.03`).

- Rationale: Uzbek-only objective no longer requires preserving multilingual priors.
- Expected impact: high; partial FT leaves lower encoder priors frozen.
- Risk: 104h USC is smaller and cleaner than Rubai's corpus, so overfitting risk is higher. Use validation WER, save best model, and compare every epoch.

### 6. Robust audio preprocessing for collected data

Rubai resamples to 16k, trims leading/trailing silence with `top_db=20`, and peak-normalizes to 0.9.

- Rationale: source heterogeneity will otherwise dominate training noise.
- Expected impact: high when scaling beyond USC.
- Risk: aggressive trimming can remove low-volume speech; keep original files and store preprocessing metadata.

### 7. Long audio segmentation heuristics

Rubai's splitter targets 10s chunks, max 25s, uses 1s padding, 0.3s silence minimum, adaptive RMS thresholds around `-32 dB`, and fallback energy segmentation.

- Rationale: pseudo-labeling and Whisper training both degrade on poor boundaries.
- Expected impact: high for YouTube/podcast/meeting ingestion.
- Risk: fixed dB thresholds will not generalize; make thresholds data-driven and audit sample cuts.

## Medium Impact, Consider

### 1. Controlled audio augmentation

Rubai's notebook contains augmentation ideas: time stretch, pitch shift, low-pass, background noise, Gaussian noise, and room simulation.

- Rationale: improves robustness to noisy production audio.
- Expected impact: medium on enterprise audio, low or negative on clean USC test.
- Risk: can hurt clean-read accuracy if applied too broadly. Use only in later domain-robustness experiments.

### 2. Evaluation speed optimizations

Rubai enables mixed precision, large teacher eval batches, TF32, cudnn benchmark, and optional `torch.compile`.

- Rationale: useful for large-scale teacher scoring and evaluation throughput.
- Expected impact: medium for iteration speed, not directly WER.
- Risk: `torch.compile` can add startup overhead and opaque failures.

### 3. Deployment chunking logic

Their bot splits long Telegram audio by silence, with fixed 25s fallback chunks.

- Rationale: useful for production inference handling of long user uploads.
- Expected impact: medium for user-facing latency and reliability.
- Risk: not directly relevant to training quality.

## Low Impact or Ignore

### 1. Do not copy weak forced-decoding behavior

`whisper_mid_demo.py` computes forced decoder IDs but does not pass them into generation. Training code also clears `forced_decoder_ids` while setting language/task.

- Rationale: our observed problem is Turkish/Kazakh leakage. We should force Uzbek decoding in evaluation/inference until ablations prove otherwise.
- Expected impact: avoids language-prior errors.
- Risk: could hurt code-switch output, but Uzbek-only WER/CER is the priority.

### 2. Hardcoded local paths and service tokens

Rubai scripts are useful references but not production-grade MLOps templates.

- Rationale: paths/tokens are environment-specific.
- Expected impact: none.
- Risk: copying would reduce reproducibility.

### 3. Exact destructive transcript regex

Rubai's normalizer is strong, but the final character whitelist is too narrow for enterprise code-switching and names.

- Rationale: use canonical Uzbek Latin, but do not blindly erase meaningful non-Uzbek spans.
- Expected impact: avoids training/eval label damage.
- Risk: less aggressive cleanup may leave more variants; handle through configurable modes.

## Recommended Adoption Order

1. Use our new Uzbek normalizer as the canonical metric/training text layer, borrowing Rubai number and Cyrillic logic where safe.
2. Run full fine-tuning with BF16, warmup ratio, weight decay, max grad norm, and SpecAugment.
3. Build pseudo-label ingestion for 500-1500h diverse Uzbek audio.
4. Score all new data with teacher ASR, normalized WER/CER, fuzzy similarity, duration/text heuristics, and manual review bands.
5. Keep forced Uzbek decoding in all Uzbek evaluation and inference.

