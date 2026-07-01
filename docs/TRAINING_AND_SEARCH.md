# Training and Search

Last rebuilt: `2026-07-01T04:50:03Z`.

## Active Stage 1 Gold+Silver Run

- Config: `configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml`.
- Output: `outputs_stage1_gold_silver_nocache/`.
- Tmux: `whisper_stage1_gold_silver_nocache`.
- Latest logged step at rebuild: `80` / `21339` (0.375%).
- Latest train metrics: `{'step': 80, 'loss': 15.000897216796876, 'grad_norm': 64.79615020751953, 'learning_rate': 7.403936269915651e-07, 'epoch': 0.0037490480932575714}`.
- No validation result yet for this restarted no-cache run.
- Monitor: `tail -f outputs_stage1_gold_silver_nocache/logs/stage1_gold_silver_nocache_training.log`.
- Resume after valid checkpoint: `.venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml --resume auto`.

## Active Stage 1 Config

- Model: `openai/whisper-large-v3`.
- Frozen: encoder layers 0-7.
- Trainable: encoder layers 8-31 and decoder.
- Optimizer: AdamW with grouped LRs; frozen params are absent from optimizer groups.
- LR groups: encoder 8-15 `2e-5`, encoder 16-23 `2e-5`, encoder 24-31 `2e-5`, decoder `2e-5`.
- Batch: 4 x gradient accumulation 8 = effective batch 32.
- Scheduler: cosine, warmup ratio 0.10. LR values are peak LRs after warmup, then cosine-decayed.
- Precision: BF16, gradient checkpointing disabled.
- Eval/save cadence: 1000 optimizer steps.
- Best model criterion: validation WER, lower is better.
- Dataset cache: disabled for persistent feature tensors; `src/train.py` uses lazy on-the-fly feature extraction.

## Completed Full-Gold Run

- Config: `configs/full_training/gold_bcd_decoder_2e5.yaml`.
- Dataset: `data/gold_master_training_schema`, 186.40h train and 10.36h validation.
- Finished one epoch: final step 5380.
- Best checkpoint: step 5000, validation WER `14.50%`, CER `3.67%`.
- Final step 5380 validation WER 14.52%, CER 3.69%; best remained step 5000.
- Peak VRAM from run metrics: `35789.31201171875` MiB.
- Runtime: `44695.6534` seconds.

Validation history:

```jsonl
{"timestamp_utc": "2026-06-28T18:42:03Z", "step": 1000, "epoch": 0.18589952130873263, "val_loss": 0.3080424964427948, "val_wer": 0.2705269727347194, "val_cer": 0.06972698039156527, "is_best": true}
{"timestamp_utc": "2026-06-28T20:54:22Z", "step": 2000, "epoch": 0.37179904261746527, "val_loss": 0.2354976087808609, "val_wer": 0.21587939306471116, "val_cer": 0.05566492430927975, "is_best": true}
{"timestamp_utc": "2026-06-28T23:06:25Z", "step": 3000, "epoch": 0.5576985639261979, "val_loss": 0.1950564682483673, "val_wer": 0.18722939501501737, "val_cer": 0.0493969633199828, "is_best": true}
{"timestamp_utc": "2026-06-29T01:18:13Z", "step": 4000, "epoch": 0.7435980852349305, "val_loss": 0.16764357686042786, "val_wer": 0.15522487030463783, "val_cer": 0.039066015985437616, "is_best": true}
{"timestamp_utc": "2026-06-29T03:30:06Z", "step": 5000, "epoch": 0.9294976065436632, "val_loss": 0.15559664368629456, "val_wer": 0.14498576276475406, "val_cer": 0.03670264521530559, "is_best": true}
{"timestamp_utc": "2026-06-29T04:55:05Z", "step": 5380, "epoch": 1.0, "val_loss": 0.15512986481189728, "val_wer": 0.14520029644654212, "val_cer": 0.03692389694697752, "is_best": false}
```

## Baselines and Failed Strategies

- Raw Whisper large-v3 on Uzbek initially produced very poor metrics: WER 105.22%, CER 45.90%; language-prior confusion with Turkish/Kazakh was observed.
- Mini fine-tune improved to WER 49.61%, CER 10.94%.
- Partial FT on USC remains protected baseline: WER 20.05%, CER 5.29%.
- Full FT on USC only degraded to WER 22.22%, CER 5.66%; conclusion: full update of all layers on ~104h was worse than partial FT.

## LR Search Summary

LR search used proxy datasets under `data/lr_search/` and kept test locked out of selection. Best proxy result was `phase4x_encoder_bcd_decoder_2e5_bs4_fast`: B/C/D encoder blocks plus decoder at 2e-5, batch 4, best proxy WER 19.13%, CER 4.84%.

Top LR-search runs by validation WER:

| Experiment | Best step | WER | CER | Eval history `(step, wer, cer)` |
|---|---:|---:|---:|---|
| phase4x_encoder_bcd_decoder_2e5_bs4_fast | 1642 | 19.13% | 4.84% | [(400, 0.3110534716679968, 0.07501416124942681), (800, 0.2567837190742219, 0.06365818789954954), (1200, 0.20909816440542697, 0.05192458123162409), (1600, 0.19273743016759776, 0.04882259326194265), (1642, 0.19134078212290503, 0.04844495994389448)] |
| phase4x_encoder_b1e5_cd_decoder2e5_bs4_fast | 1600 | 19.27% | 4.83% | [(400, 0.3112529928172386, 0.07487929220726675), (800, 0.2559856344772546, 0.06233647128638092), (1200, 0.21288906624102155, 0.0616351522671486), (1600, 0.19273743016759776, 0.04825614328487039), (1642, 0.1931364724660814, 0.048714698028214605)] |
| phase4x_main_encoder_cd_decoder_5em05_bs4_fast | 1642 | 19.37% | 5.00% | [(400, 0.3375897845171588, 0.08610039651498395), (800, 0.28371907422186754, 0.07633587786259542), (1200, 0.21428571428571427, 0.05758908100234672), (1600, 0.1949321628092578, 0.05017128368354328), (1642, 0.19373503591380686, 0.0500094408329512)] |
| phase4x_blockwise_c1e5_d5e5_decoder5e5_bs4_fast | 1642 | 19.43% | 6.88% | [(400, 0.3234237829209896, 0.08396946564885496), (800, 0.26596169193934555, 0.06781215439807947), (1200, 0.21867517956903432, 0.06303779030561325), (1600, 0.19473264166001597, 0.06870229007633588), (1642, 0.19433359936153233, 0.06878321150163191)] |
| phase4x_main_encoder_cd_decoder_2em05_bs4_fast | 1642 | 19.49% | 5.09% | [(400, 0.3078611332801277, 0.0782510182612683), (800, 0.25139664804469275, 0.06217462843578885), (1200, 0.20371109337589785, 0.052059450273784155), (1600, 0.19573024740622505, 0.05098049793650365), (1642, 0.1949321628092578, 0.05089957651120762)] |
| phase4x_encoder_bcd_decoder_5e5_bs4_fast | 1642 | 20.09% | 5.14% | [(400, 0.36671987230646447, 0.09926361502980606), (800, 0.27992817238627293, 0.07312599465918593), (1200, 0.2184756584197925, 0.05607854773015402), (1600, 0.2011173184357542, 0.05138510506298384), (1642, 0.20091779728651238, 0.051412078871415856)] |
| phase4x_main_upper_encoder_decoder_2em05_bs4_fast | 1600 | 21.21% | 5.60% | [(400, 0.33978451715881886, 0.08526420845359156), (800, 0.27673583399840385, 0.07040164000755267), (1200, 0.2162809257781325, 0.057616054810778736), (1600, 0.21209098164405427, 0.05599762630485798), (1642, 0.21268954509177973, 0.056051573921722005)] |
| phase4x_encoder_bcd_decoder_1e5_bs4_fast | 1600 | 21.59% | 5.42% | [(400, 0.31424581005586594, 0.07647074690475548), (800, 0.2603750997605746, 0.06522266878860626), (1200, 0.22007182761372707, 0.05570091441210585), (1600, 0.21588188347964885, 0.054217354948345155), (1642, 0.21588188347964885, 0.05437919779893723)] |
| phase4x_full_encoder_decoder_1e5_bs1_safe | 1642 | 22.35% | 5.64% | [(400, 0.3180367118914605, 0.0795457610660049), (800, 0.2531923383878691, 0.06481806166212607), (1200, 0.24062250598563448, 0.06023251422868395), (1600, 0.22386272944932162, 0.05632131200604213), (1642, 0.22346368715083798, 0.05640223343133817)] |
| phase3_freeze_boundary_15 | 1600 | 23.66% | 6.10% | [(400, 0.32382282521947325, 0.08054379197798937), (800, 0.2605746209098164, 0.06646346397647884), (1200, 0.24920191540303271, 0.06303779030561325), (1600, 0.23663208300079808, 0.060960807056348286), (1642, 0.23703112529928172, 0.06093383324791627)] |
| phase2_upper_encoder_8em06 | 1642 | 25.40% | 6.75% | [(400, 0.361731843575419, 0.09276292719769104), (800, 0.29209896249002393, 0.07674048498907561), (1200, 0.26975259377494015, 0.07080624713403286), (1600, 0.2549880287310455, 0.06759636393062336), (1642, 0.2539904229848364, 0.06754241631375935)] |
| phase2_upper_encoder_5em06 | 1642 | 26.80% | 7.23% | [(400, 0.3741021548284118, 0.09524451757343619), (800, 0.29728651237031123, 0.07833193968656435), (1200, 0.2789305666400638, 0.07560758503493108), (1600, 0.2691540303272147, 0.07299112561702586), (1642, 0.26795690343176376, 0.07231678040622555)] |

Main interpretation:

- Decoder-only tuning was not enough on the 30h proxy.
- Unfreezing only upper encoder D helped but was weaker than B/C/D.
- Training encoder B/C/D plus decoder at 2e-5 was the best observed proxy regime.
- Full encoder training at 2e-5 collapsed badly on proxy validation and should not be used without a new controlled design.
