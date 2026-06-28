# Experiment Ledger

Chronological record of major experiments and outcomes. Metrics are WER/CER ratios, so
`0.20` means 20%.

## 1. Raw Whisper Large-v3 Baseline

- Model: `openai/whisper-large-v3`.
- Uzbek result: WER `1.0522`, CER `0.4590`.
- Failure mode: severe Uzbek errors and Turkish/Kazakh-like language-prior confusion.
- Conclusion: raw model is not usable for the target.

## 2. Mini Fine-Tune

- Model: Whisper large-v3.
- Data: small Uzbek subset.
- Result: WER `0.4961`, CER `0.1094`.
- Conclusion: fine-tuning strongly improves Uzbek quality.

## 3. Protected USC Partial Fine-Tune

- Data: USC about 104.63h.
- Strategy: encoder 0-23 frozen, encoder 24-31 + decoder trainable.
- Trainable parameters: 1,063,930,880.
- Epochs: 1.
- Result: test WER `0.2005258480`, CER `0.0529079419`.
- Path: `models/partial_ft_usc_baseline/`.
- Conclusion: best completed locked-test model; protected baseline.

## 4. USC Full Fine-Tune

- Data: USC about 104.63h.
- Strategy: all 1,543,490,560 parameters trainable.
- Precision: BF16.
- Encoder LR: `2e-6`.
- Decoder LR: `8e-6`.
- Epochs: 1.
- Result: test WER `0.2221522737`, CER `0.0565825834`.
- Conclusion: failed promotion; lower encoder should not be aggressively updated on
  USC-sized data without further evidence.

## 5. Gold Corpus Construction

- Sources retained as Gold: USC, Common Voice Uzbek, FLEURS Uzbek.
- Final Gold: 184,140 rows / 207.1150h.
- Splits: train 186.4037h, validation 10.3556h, test 10.3557h.
- Validation: no missing paths, path/hash leakage 0, known speaker leakage 0.
- FeruzaSpeech was removed from Gold and moved to train-only Silver because of
  restrictive/gated terms.

## 6. LR-Search Framework and Proxies

- Coarse proxy: 9.9971h train.
- Main proxy: 29.9987h train.
- Composition: 50% USC, 40% Common Voice, 10% FLEURS.
- Test loading/evaluation disabled during search.
- Leakage audit passed.

## 7. Decoder-Only LR Search

Decoder-only helped but was insufficient. Important results:

| Experiment | Decoder LR | Dataset | Best WER | Best CER | Conclusion |
|---|---:|---|---:|---:|---|
| `phase1b_decoder_lr_8e6` | 8e-6 | coarse | 0.440624 | 0.116689 | best coarse decoder-only |
| `phase2_decoder_8em06` | 8e-6 | main | 0.492418 | 0.138403 | weak control |
| `phase1a_decoder_lr_5e5` | 5e-5 | coarse screen | 2.640144 | 2.144167 | too aggressive |
| `phase2_decoder_2em05` | 2e-5 | main | 6.313448 | 2.530062 | degraded badly |

Conclusion: decoder-only adaptation cannot reach target quality and high decoder LR
without encoder adaptation is unsafe.

## 8. Upper-Encoder LR Search

Encoder 24-31 + decoder improved over decoder-only.

| Experiment | Encoder LR | Decoder LR | WER | CER |
|---|---:|---:|---:|---:|
| `phase2_upper_encoder_5em07` | 5e-7 | 8e-6 | 0.311652 | 0.082486 |
| `phase2_upper_encoder_1em06` | 1e-6 | 8e-6 | 0.299082 | 0.079492 |
| `phase2_upper_encoder_2em06` | 2e-6 | 8e-6 | 0.283919 | 0.075581 |
| `phase2_upper_encoder_5em06` | 5e-6 | 8e-6 | 0.267957 | 0.072317 |
| `phase2_upper_encoder_8em06` | 8e-6 | 8e-6 | 0.253990 | 0.067542 |

Conclusion: more encoder adaptation was beneficial on the proxy.

## 9. Freeze Boundary and Blockwise Search

Important 30h proxy results:

| Experiment | Trainable encoder | LR schedule | WER | CER |
|---|---|---|---:|---:|
| `phase3_freeze_boundary_15` | 16-31 | enc `5e-6`, dec `8e-6` | 0.236632 | 0.060961 |
| `phase4x_main_upper_encoder_decoder_2em05_bs4_fast` | 24-31 | enc/dec `2e-5` | 0.212091 | 0.055998 |
| `phase4x_main_encoder_cd_decoder_2em05_bs4_fast` | 16-31 | enc/dec `2e-5` | 0.194932 | 0.050900 |
| `phase4x_main_encoder_cd_decoder_5em05_bs4_fast` | 16-31 | enc/dec `5e-5` | 0.193735 | 0.050009 |
| `phase4x_encoder_b1e5_cd_decoder2e5_bs4_fast` | 8-31 | B `1e-5`, C/D/dec `2e-5` | 0.192737 | 0.048256 |
| `phase4x_encoder_bcd_decoder_2e5_bs4_fast` | 8-31 | B/C/D/dec `2e-5` | 0.191341 | 0.048445 |

Conclusion: best proxy WER came from freezing encoder 0-7 and training encoder 8-31 +
decoder at `2e-5`.

## 10. Current Full Gold Promotion Run

- Started: 2026-06-28 16:30 UTC.
- tmux: `whisper_gold_ft`.
- Config: `configs/full_training/gold_bcd_decoder_2e5.yaml`.
- Strategy: encoder 0-7 frozen; encoder 8-31 + decoder `2e-5`.
- Data: full Gold training schema.
- Steps: 5,380.
- Status at 2026-06-28T16:49:59Z: running around step 253; no validation yet.

Conclusion: pending. This run determines whether the best proxy schedule transfers to
the full Gold corpus.
