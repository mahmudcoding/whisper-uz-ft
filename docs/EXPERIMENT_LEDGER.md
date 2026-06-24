# Experiment Ledger

**Document role:** Chronological record of experiments and evidence.  
**Rule:** Append results; do not rewrite history. Current operational state belongs in
`STATUS.md`.

## Metric Conventions

- WER/CER are ratios unless explicitly shown as percentages.
- Metrics are comparable only when dataset, split, normalization, and decoding policy
  match.
- Search-proxy validation metrics are not directly comparable to protected USC test
  metrics.

## E001 - Raw Whisper Baseline

**Model:** `openai/whisper-large-v3`  
**Decoding:** forced Uzbek transcription  
**Artifact:** `outputs/baseline_metrics.json`

| Split | Samples | WER | CER |
|---|---:|---:|---:|
| mini validation | 375 | 1.379917 | 0.810883 |
| mini test | 279 | 1.052247 | 0.459004 |

Observed errors:

- Turkish/Kazakh-like spellings;
- wrong-script output;
- repeated-token hallucinations;
- severe insertions.

Conclusion: raw large-v3 is not deployable for Uzbek.

## E002 - Mini Fine-Tune

**Config:** `configs/mini_train.yaml`  
**Output:** `outputs/mini/`  
**Epochs:** 2

| Metric | Value |
|---|---:|
| test loss | 0.589842 |
| test WER | 0.496067 |
| test CER | 0.109443 |

Conclusion: even small supervised adaptation produces a major Uzbek improvement.

## E003 - Partial Fine-Tune on USC

**Registry ID:** `partial_ft_usc_baseline`  
**Protected archive:** `archive/partial_ft_usc/`  
**Dataset:** USC, 104.63h  
**Config snapshot:** `archive/partial_ft_usc/config/`

Training:

- encoder 0-23 frozen;
- encoder 24-31 trainable;
- decoder trainable;
- 1,063,930,880 trainable parameters;
- FP16;
- one epoch;
- effective batch 32;
- LR `1e-5`.

| Metric | Value |
|---|---:|
| test loss | 0.227584 |
| test WER | **0.200526** |
| test CER | **0.052908** |

Conclusion: current best completed model and protected baseline.

## E004 - Full-FT BF16 Dry Run

**Config:** `configs/full_ft_dry_run.yaml`  
**Output:** `outputs_full_ft_dry_run/`  
**Steps:** 100

Evidence:

- all large-v3 parameters trainable;
- BF16 and gradient checkpointing worked on A40;
- peak VRAM approximately 30 GiB;
- loss decreased from approximately 28.31 to 5.269;
- checkpoint save and evaluation completed.

Conclusion: full-FT execution is feasible on the hardware, but feasibility does not
imply quality superiority.

## E005 - One-Epoch Full FT on USC

**Config:** `configs/full_ft_uzbek.yaml`  
**Output:** `outputs_full_ft/`  
**Completion:** 2026-06-24 01:48 UTC  
**Steps:** 3,114

Training:

- 1,543,490,560 trainable parameters;
- BF16;
- encoder LR `2e-6`;
- decoder LR `8e-6`;
- effective batch 32;
- cosine schedule;
- gradient clipping 1.0;
- one epoch.

| Metric | Value |
|---|---:|
| test loss | 0.243817 |
| test WER | 0.222152 |
| test CER | 0.056583 |
| hallucination rate | 0 |
| language-confusion rate | 0 |

Conclusion: full FT removed measured confusion but worsened WER/CER relative to E003.
Lower encoder adaptation is not justified on USC-only evidence.

Operational event:

- the run was originally launched for four epochs;
- requirement changed to one epoch;
- checkpoint resume initially failed with PyTorch 2.5.1;
- environment was upgraded to PyTorch 2.7.1+cu126;
- optimizer/scheduler/global-step resume then succeeded.

## E006 - Gold Master Construction

**Output:** `data/gold_master/`  
**Reports:** `reports/gold_quality_report/`,
`reports/gold_dedup_report/`

| Field | Value |
|---|---:|
| raw rows | 184,325 |
| raw hours | 207.2744 |
| quality rejects | 50 |
| audio duplicates removed | 135 |
| final rows | 184,140 |
| final hours | 207.1150 |

Validation found zero missing paths, exact path leakage, content-hash leakage, and known
speaker leakage.

Conclusion: the project has a substantially larger clean corpus ready for schema
conversion and post-search training.

## E007 - Inference Smoke Benchmark

**Engine:** faster-whisper  
**Model:** large-v3  
**Precision:** FP16  
**Beam:** 1

Initial smoke result:

- RTF `0.1796`;
- speed `5.57x`;
- peak VRAM 4,257 MB.

Conclusion: useful framework validation, insufficient for production capacity.

## E008 - A40 Batch Scaling

**Report:** `benchmark/reports/faster_whisper_batch_scaling_a40.md`

Best smoke batch: 2.

- RTF `0.0998`;
- speed `10.02x`;
- peak VRAM 4,257 MB.

Conclusion: batch 2 was best on smoke audio, but workload size was too small for
enterprise sizing.

## E009 - Long-Form Offline Capacity

**Report:** `benchmark/reports/long_form_offline_capacity_report.md`  
**Dataset:** 5h USC-derived long-form audio

Best measured configuration:

- faster-whisper;
- large-v3;
- FP16;
- beam 1;
- batch 4;
- end-to-end RTF `0.0230`;
- throughput `43.82 audio-hours/hour/GPU`;
- peak VRAM 5,089 MB.

Conclusion: use this result, not smoke throughput, for current offline A40 planning.
Beam 5 remains unmeasured for the full run.

## E010 - LR Search Infrastructure and Leakage Audit

**Date:** 2026-06-24  
**Plan:** `TRAINING_AND_SEARCH.md`  
**Audit:** `reports/lr_search/data_leakage_audit.md`

Implemented:

- deterministic 10h and 30h proxies;
- explicit tuning modes;
- inherited configs;
- test-loading prohibition;
- GPU telemetry;
- autonomous phase controller;
- validation-only comparison.

Audit status: pass.

## E011 - Phase 1A Decoder LR `2e-6`

**Experiment ID:** `phase1a_decoder_lr_2e6`  
**Output:** `outputs_lr_search/phase1a_decoder_lr_2e6/`  
**Mode:** decoder-only  
**Steps:** 300  
**Dataset:** coarse proxy

| Metric | Step 150 | Step 300 |
|---|---:|---:|
| validation loss | 1.617290 | 1.427962 |
| validation WER | 0.662335 | **0.644542** |
| validation CER | 0.166994 | **0.161095** |
| hallucination | 0 | 0 |
| language confusion | 0.008284 | 0.008284 |

Runtime: 4,860 seconds. Peak VRAM: 21,471 MiB.

Decision: stable Phase 1A survivor.

## E012 - Phase 1A Decoder LR `8e-6`

**Experiment ID:** `phase1a_decoder_lr_8e6`  
**Status at documentation rebuild:** training reached step 300; final validation running  
**Output:** `outputs_lr_search/phase1a_decoder_lr_8e6/`

Midpoint validation at step 150:

| Metric | Value |
|---|---:|
| validation loss | 0.6979 |
| validation WER | 0.5398 |
| validation CER | 0.1421 |
| hallucination rate | 0.001183 |
| language-confusion rate | 0.001183 |

Interim interpretation: materially better than `2e-6` at the same step, with no
stability rejection. Final decision awaits step-300 metrics.

## Pending Experiments

- Phase 1A: decoder `2e-5`, `5e-5`.
- Phase 1B: full coarse runs for survivors.
- Phase 2A: top-two decoder confirmation on 30h.
- Phase 2B: upper encoder LR `5e-7`, `1e-6`, `2e-6`, `5e-6`.
- Phase 3: freeze-boundary comparison.
- Final locked proxy test.
- One-epoch 207h Gold training.
