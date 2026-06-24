# Current Status

**Last verified:** 2026-06-24 08:54 UTC  
**Purpose:** Answer "What is true and running now?"  
**Update rule:** Refresh after every job transition, completed experiment, blocker, or
change to the best model.

## Executive Summary

- Current best completed model: `partial_ft_usc_baseline`.
- Best test metrics: WER `0.2005258480`, CER `0.0529079419`.
- Protected model path: `archive/partial_ft_usc/model/`.
- Gold corpus: `207.1150h`, built from USC, Common Voice Uzbek, and FLEURS Uzbek.
- Active work: autonomous learning-rate and freeze-boundary search.
- Test-leakage audit: passed.
- Full FT on USC completed but did not beat partial FT.

## Active Runtime

### Autonomous LR Search

| Field | Value |
|---|---|
| tmux session | `whisper_lr_search` |
| controller | `scripts/lr_search/autonomous_search.py` |
| console log | `reports/lr_search/autonomous_search_console.log` |
| event log | `reports/lr_search/autonomous_search.log` |
| output root | `outputs_lr_search/` |
| current phase | Phase 1A: 300-step decoder-LR divergence screen |
| current experiment | `phase1a_decoder_lr_8e6` |
| current operation | step-300 final validation in progress |
| current dataset | `data/lr_search/coarse_10h/` |
| test loading | disabled |
| test evaluation | disabled |

Current process state must be rechecked with:

```bash
tmux has-session -t whisper_lr_search
pgrep -af 'autonomous_search.py|run_experiment.py|src/train.py'
tail -40 reports/lr_search/autonomous_search_console.log
nvidia-smi
```

### Completed Phase 1A Result

`phase1a_decoder_lr_2e6` completed 300 optimizer steps and passed stability screening.

| Metric | Value |
|---|---:|
| Best validation step | 300 |
| Validation loss | 1.427962 |
| Validation WER | 0.644542 |
| Validation CER | 0.161095 |
| Hallucination rate | 0 |
| Language-confusion rate | 0.008284 |
| Peak VRAM | 21,471 MiB |
| Runtime | 4,860 seconds |

These proxy metrics are for LR ranking only. They are not comparable to the protected
USC test baseline because the dataset composition and evaluation split differ.

For `phase1a_decoder_lr_8e6`, training reached step 300 without a safety stop. The
step-300 validation was processing the 845-row validation manifest at the last
verification. Midpoint metrics were WER `0.5398`, CER `0.1421`, hallucination
`0.001183`, and language confusion `0.001183`.

## Completed Training

### Protected Partial FT Baseline

- Base: `openai/whisper-large-v3`.
- Dataset: USC, 104.63h.
- Frozen: encoder 0-23.
- Trainable: encoder 24-31 and decoder.
- Trainable parameters: 1,063,930,880.
- Precision: FP16.
- Epochs: 1.
- Test WER: `0.2005258480`.
- Test CER: `0.0529079419`.

### USC Full FT

- All 1,543,490,560 parameters trainable.
- BF16, gradient checkpointing.
- Encoder LR `2e-6`, decoder LR `8e-6`.
- One epoch, 3,114 steps.
- Final model: `outputs_full_ft/final_model/`.
- Test WER: `0.2221522737`.
- Test CER: `0.0565825834`.
- Hallucination and language-confusion rates: 0.

Conclusion: full FT reduced measured confusion but degraded aggregate WER/CER. Do not
extend USC-only full FT without new evidence.

## Data State

| Corpus | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| USC project splits | 99,617 rows | 3,762 | 3,821 | 104.63h |
| Gold master | 186.4037h | 10.3556h | 10.3557h | 207.1150h |
| LR coarse proxy | 9.9971h | 0.9988h | 0.9992h | 11.9951h |
| LR main proxy | 29.9987h | 1.0002h | 1.0002h | 31.9991h |

Gold validation status:

- missing audio paths: 0;
- exact path leakage: 0;
- exact content-hash leakage: 0;
- known speaker leakage: 0.

FeruzaSpeech remains blocked by gated Hugging Face access.

## Current Risks

1. Proxy WER has high sampling variance; small deltas must be treated as ties.
2. FLEURS speaker identity is unavailable, so speaker isolation cannot be proven there.
3. Gold quality scoring is mostly heuristic; full teacher-agreement scoring is pending.
4. Gold master schema differs from the current training schema.
5. `outputs_lr_search/` is large and growing; monitor disk before long phases.

## Immediate Sequence

The autonomous controller should:

1. complete Phase 1A for decoder LRs `2e-6`, `8e-6`, `2e-5`, `5e-5`;
2. reject unstable candidates;
3. run full coarse Phase 1B for survivors;
4. promote the top two, respecting practical tie thresholds;
5. confirm both on the 30h proxy;
6. search upper-encoder LR;
7. compare decoder-only, encoder 24-31 + decoder, and encoder 16-31 + decoder;
8. generate `reports/lr_search/FINAL_RECOMMENDATION.md`.

Do not start a second controller or manually consume the test split.
