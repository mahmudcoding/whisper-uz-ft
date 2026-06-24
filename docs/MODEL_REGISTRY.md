# Model Registry

**Document role:** Locate decision-relevant models and state their provenance,
evaluation, and retention policy.

## Registry Rules

Every model used for comparison, promotion, deployment, or recovery must record:

- stable registry ID;
- base model;
- training data and split policy;
- tuning mode and precision;
- config/output paths;
- metrics and evaluation set;
- lifecycle state;
- retention policy.

Do not treat an arbitrary checkpoint directory as a registered model.

## Registered Models

### `raw_whisper_large_v3`

| Field | Value |
|---|---|
| Base | `openai/whisper-large-v3` |
| Training | none |
| Local path | Hugging Face cache |
| Evaluation | USC mini test |
| WER/CER | 1.052247 / 0.459004 |
| State | reference baseline |
| Retention | external/re-downloadable |

Use only as an initialization and baseline. It is not production-quality Uzbek ASR.

### `mini_ft_usc`

| Field | Value |
|---|---|
| Base | `openai/whisper-large-v3` |
| Data | USC mini splits |
| Config | `configs/mini_train.yaml` |
| Model | `outputs/mini/final_model/` |
| Metrics | `outputs/mini/test_metrics.json` |
| WER/CER | 0.496067 / 0.109443 |
| State | completed smoke model |
| Retention | useful for pipeline tests |

### `partial_ft_usc_baseline`

| Field | Value |
|---|---|
| Base | `openai/whisper-large-v3` |
| Data | USC, 104.63h |
| Tuning | encoder 24-31 + decoder |
| Precision | FP16 |
| Epochs | 1 |
| Model | `archive/partial_ft_usc/model/` |
| Metrics | `archive/partial_ft_usc/metrics/test_metrics.json` |
| WER/CER | **0.200526 / 0.052908** |
| State | current best completed model |
| Retention | **protected; never modify or delete** |

This is the promotion threshold for future models.

### `full_ft_usc_layerwise`

| Field | Value |
|---|---|
| Base | `openai/whisper-large-v3` |
| Data | USC, 104.63h |
| Tuning | full FT |
| Precision | BF16 |
| Encoder/decoder LR | `2e-6` / `8e-6` |
| Config | `configs/full_ft_uzbek.yaml` |
| Model | `outputs_full_ft/final_model/` |
| Final checkpoint | `outputs_full_ft/checkpoint-3114/` |
| Metrics | `outputs_full_ft/test_metrics.json` |
| WER/CER | 0.222152 / 0.056583 |
| State | completed, not promoted |
| Retention | retain for error analysis and full-FT control |

### `lr_search_phase1a_decoder_2e6`

| Field | Value |
|---|---|
| Base | `openai/whisper-large-v3` |
| Data | coarse LR proxy |
| Tuning | decoder-only |
| Decoder LR | `2e-6` |
| Output | `outputs_lr_search/phase1a_decoder_lr_2e6/` |
| Best validation WER/CER | 0.644542 / 0.161095 |
| State | completed Phase 1A survivor |
| Retention | retain until search completion |

This is a search artifact, not a production candidate.

### `lr_search_phase1a_decoder_8e6`

| Field | Value |
|---|---|
| Base | `openai/whisper-large-v3` |
| Data | coarse LR proxy |
| Tuning | decoder-only |
| Decoder LR | `8e-6` |
| Output | `outputs_lr_search/phase1a_decoder_lr_8e6/` |
| Midpoint validation WER/CER | 0.5398 / 0.1421 |
| State | step-300 final validation running at documentation rebuild |
| Retention | retain until search completion |

## Planned Registry Entries

- locked LR-search winner;
- one-epoch Gold master model;
- Gold + Silver curriculum models;
- production-domain adaptation model;
- CTranslate2/faster-whisper conversion of the promoted model.

## Promotion Criteria

A model can replace `partial_ft_usc_baseline` only if:

1. configuration was selected without test leakage;
2. final evaluation uses the locked comparable test set;
3. WER improves materially, or a statistically tied WER has clear CER/domain benefits;
4. hallucination and language confusion do not regress materially;
5. model provenance and config are reproducible;
6. checkpoint and final model are verified;
7. the registry and experiment ledger are updated.
