# Model Registry

Last rebuilt: `2026-07-01T04:50:03Z`.

| Model ID | Path | Data | Training regime | Status | Metrics |
|---|---|---|---|---|---|
| `partial_ft_usc_baseline` | `models/partial_ft_usc_baseline/model/` | USC ~104h | encoder 0-23 frozen, encoder 24-31 + decoder trainable | Protected baseline | test WER 20.05%, CER 5.29% |
| `full_ft_usc_failed` | `outputs_full_ft/` | USC ~104h | full FT all params | Not recommended | test WER 22.22%, CER 5.66% |
| `full_gold_bcd_decoder_2e5` | `outputs_full_gold/best_model/` | Gold train 186.40h | encoder A frozen, B/C/D + decoder at 2e-5 | Best completed Gold-only model | val WER 14.50%, CER 3.67% at step 5000 |
| `stage1_gold_silver_bcd_decoder_2e5_nocache` | `outputs_stage1_gold_silver_nocache/` | Gold+Silver train 981.76h | encoder A frozen, B/C/D + decoder at 2e-5 | Running | no validation yet after restart |

Protected artifacts:

- `models/partial_ft_usc_baseline/` must not be modified or deleted.
- `outputs_full_gold/best_model/` is the best completed Gold-only model and should be preserved.
