# Roadmap

**Document role:** Prioritized, actionable work.  
**Priority definitions:** P0 blocks the current objective; P1 is the next quality
milestone; P2 expands data/model capability; P3 productionizes the winner.

## P0 - Complete LR and Freeze-Boundary Search

1. Finish Phase 1A decoder screens.
2. Run Phase 1B for stable candidates.
3. Promote top two using validation WER/CER and practical tie thresholds.
4. Confirm decoder LRs on the 30h proxy.
5. Search encoder 24-31 LR.
6. Compare decoder-only, 24-31 + decoder, and 16-31 + decoder.
7. Produce `reports/lr_search/FINAL_RECOMMENDATION.md`.
8. Verify test manifests remain unchanged.

Exit criterion: decoder LR, encoder LR, and freeze boundary selected from stable
validation evidence.

## P1 - Train the 207.12h Gold Corpus

1. Use the validated training schema at `data/gold_master_training_schema/`.
2. Integrate or verify trust/source-aware sampling.
3. Run manifest, audio, normalization, leakage, and one-forward-pass checks.
4. Launch one epoch using the LR-search winner.
5. Select checkpoints on Gold validation only.
6. Evaluate the locked Gold test once.
7. Compare against the protected partial-FT baseline.
8. Register and archive any promoted model.

Exit criterion: reproducible Gold model with measured comparable test metrics.

## P1 - Paired Error Analysis

Compare:

- protected partial FT;
- USC full FT;
- Gold winner.

Classify:

- acoustic substitutions;
- Uzbek language-prior errors;
- script/normalization mismatches;
- Turkish/Kazakh confusion;
- Russian/English code-switch errors;
- hallucinations and omissions.

Exit criterion: quantified error categories that determine the next data/training
investment.

## Completed - FeruzaSpeech Silver Reclassification

- Prepared 12,855 clips / 57.8296h from the local ZIP.
- Rejected 136 unaligned clips longer than 30 seconds.
- Rejected one additional low-information `18+` sample during quality scoring.
- Originally integrated 12,854 clips / 57.8279h into Gold, then reclassified them
  to train-only Silver on 2026-06-27 because of restrictive gated terms.
- Verified zero path, content-hash, and reliable-speaker leakage.
- Removed the superseded pre-Feruza copy after validating the new corpus.

The gated license permits academic research/internal use and prohibits redistribution.
Commercial use remains a legal/governance review item.

## P2 - Teacher-Assisted Quality Scoring

Use the strongest verified Uzbek model to score representative and then full-corpus
agreement:

- teacher WER/CER;
- transcript similarity;
- hallucination;
- duration/text plausibility;
- audio quality.

Calibrate keep/suspicious/reject thresholds with human review.

## P2 - Silver Data Acquisition

Order:

1. UzbekVoice filtered;
2. IT YouTube;
3. news YouTube;
4. Tashkent podcasts.

Required before training:

- license/source record;
- Gold overlap removal;
- normalization;
- teacher agreement;
- audio quality;
- domain and speaker split controls.

## P2 - Missing Production Domains

Highest-impact collection:

- telephony and call centers;
- meetings and interruptions;
- spontaneous conversation;
- regional dialects;
- Uzbek-Russian code-switching;
- elderly and children speech;
- far-field/noisy rooms.

## P2 - Curriculum Training

1. Gold only.
2. Gold + filtered Silver.
3. Gold + Silver + high-confidence Bronze.
4. Low-LR production-domain adaptation.

Run sampling-weight and curriculum ablations. Do not assume initial 4.0/1.5/1.0 weights
are optimal.

## P3 - Evaluation Expansion

- fixed domain-specific benchmark sets;
- long-form meeting and call tests;
- normalized and raw WER/CER;
- confidence intervals/bootstrap comparisons;
- public model comparisons on identical audio;
- code-switch and language-confusion suites.

## P3 - Inference Productionization

1. Convert the winner to CTranslate2.
2. Verify quality parity.
3. Re-run 5h long-form benchmarks.
4. Benchmark beam 1 and guarded beam 5.
5. Measure streaming separately.
6. Update costs with current cloud/self-hosted prices.
7. Add queueing, autoscaling, observability, and service overhead.

## Deferred

- Distillation or smaller models: evaluate after the large-v3 quality ceiling is known.
- Shallow-fusion LM/rescoring: consider if language-prior errors remain after data scale.
- Text-only decoder adaptation: research path, not current production plan.
- Multilingual retention: out of scope for this model.
