# Model Registry

This registry distinguishes locked-test models from proxy-search artifacts. Do not
promote a proxy result without a final evaluation protocol.

## Protected Models

### `partial_ft_usc_baseline`

| Field | Value |
|---|---|
| Path | `models/partial_ft_usc_baseline/model/` |
| Metrics | `models/partial_ft_usc_baseline/metrics/test_metrics.json` |
| Base | `openai/whisper-large-v3` |
| Dataset | USC only |
| Strategy | encoder 0-23 frozen; encoder 24-31 + decoder trainable |
| Precision | FP16 |
| Epochs | 1 |
| Test WER | `0.2005258480` |
| Test CER | `0.0529079419` |
| Status | protected, immutable |

This is the best completed locked-test model at the time of this registry update.

## Active Candidate

### `full_gold_bcd_decoder_2e5`

| Field | Value |
|---|---|
| Config | `configs/full_training/gold_bcd_decoder_2e5.yaml` |
| Output | `outputs_full_gold/` |
| Base | `openai/whisper-large-v3` |
| Dataset | full Gold only, `data/gold_master_training_schema/` |
| Strategy | encoder 0-7 frozen; encoder 8-31 + decoder trainable |
| LR | B/C/D/decoder `2e-5` |
| Precision | BF16 |
| Batch / accumulation | 4 / 8 |
| Epochs | 1 |
| Test loading | disabled |
| Status | running in tmux session `whisper_gold_ft` |

This candidate is based on the best 30h proxy LR-search result. It is not promoted
until validation trajectory and final locked evaluation are complete.

## Completed Negative / Historical Models

### `usc_full_ft_1epoch`

| Field | Value |
|---|---|
| Metrics | `outputs_full_ft/test_metrics.json` |
| Base | `openai/whisper-large-v3` |
| Dataset | USC only |
| Strategy | full fine-tune, all parameters trainable |
| Precision | BF16 |
| Encoder LR | `2e-6` |
| Decoder LR | `8e-6` |
| Test WER | `0.2221522737` |
| Test CER | `0.0565825834` |
| Status | failed promotion |

Conclusion: full FT on USC degraded relative to partial FT.

## Proxy Search Artifacts

Best 30h proxy artifact:

| Experiment | Path | WER | CER | Status |
|---|---|---:|---:|---|
| `phase4x_encoder_bcd_decoder_2e5_bs4_fast` | `outputs_lr_search/phase4x_encoder_bcd_decoder_2e5_bs4_fast/` | `0.1913407821` | `0.0484449599` | selected for full Gold test |
| `phase4x_encoder_b1e5_cd_decoder2e5_bs4_fast` | `outputs_lr_search/phase4x_encoder_b1e5_cd_decoder2e5_bs4_fast/` | `0.1927374302` | `0.048256` | close runner-up |
| `phase4x_full_encoder_decoder_2e5_bs1_safe` | `outputs_lr_search/phase4x_full_encoder_decoder_2e5_bs1_safe/` | `3.911213` | `2.254956` | rejected |

Many old LR-search checkpoint directories were removed during storage cleanup. Metrics
and logs remain the historical record; not every listed output is resumable.
