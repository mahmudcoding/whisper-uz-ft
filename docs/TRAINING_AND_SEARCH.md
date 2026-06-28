# Training and Learning-Rate Search

This file documents the actual training pipeline, current best evidence, and how to
continue safely.

## Entry Points

| Purpose | File |
|---|---|
| Main trainer | `src/train.py` |
| Model loading/freezing | `src/model.py` |
| Active full Gold config | `configs/full_training/gold_bcd_decoder_2e5.yaml` |
| Legacy partial FT config | `configs/train.yaml` |
| USC full-FT config | `configs/full_ft_uzbek.yaml` |
| LR-search configs | `configs/lr_search/` |
| LR experiment runner | `scripts/lr_search/run_experiment.py` |
| LR comparison | `scripts/lr_search/compare_experiments.py` |
| Freeze verifier | `scripts/lr_search/verify_freeze_modes.py` |

## Model Contract

All experiments are Uzbek-only:

```yaml
model_name: openai/whisper-large-v3
language: uz
task: transcribe
```

`src/model.py` rejects non-`uz`/non-`transcribe` configuration and sets forced decoder
prompt IDs. Automatic language detection is not part of model selection.

## Tuning Modes and Parameter Counts

Whisper large-v3 has 32 encoder layers.

| Mode | Encoder state | Decoder state | Trainable parameters |
|---|---|---|---:|
| `decoder_only` | all frozen | trainable | 906,521,600 |
| `encoder_24_31_plus_decoder` | 0-23 frozen | trainable | 1,063,930,880 |
| `encoder_16_31_plus_decoder` | 0-15 frozen | trainable | 1,221,340,160 |
| `blockwise_encoder_lr` | per-block LR controls | trainable if decoder LR > 0 | schedule-dependent |
| `full` / `full_ft` | all trainable | trainable | 1,543,490,560 |

Blockwise groups:

| Config key | Layers |
|---|---:|
| `encoder_block_a_lr` | 0-7 |
| `encoder_block_b_lr` | 8-15 |
| `encoder_block_c_lr` | 16-23 |
| `encoder_block_d_lr` | 24-31 |

For blockwise configs, LR `0` freezes that block. Startup logs print actual frozen
state, LR, trainable parameters, and total trainable percentage. Trust that startup
report over a config comment.

## Optimizer, Scheduler, and Precision

`src/train.py` builds custom AdamW parameter groups:

- per-block encoder LRs for `blockwise_encoder_lr`;
- decoder/projection LR from `decoder_learning_rate`;
- weight decay for standard weights;
- no decay for bias and layer norm parameters;
- tied/shared parameters are deduplicated.

The scheduler is the HuggingFace Trainer scheduler named by config key `scheduler`.
The active full Gold run uses:

```yaml
scheduler: cosine
warmup_ratio: 0.10
warmup_steps: 0
```

Configured LRs are peak/base LRs after warmup, not constant whole-run values. The
active full Gold log showed LR `7.063e-7` at step 20 and `6.654e-6` at step 180 during
warmup toward `2e-5`.

Current precision default for large-v3 training is BF16 on A40. The active full Gold
run disables gradient checkpointing because batch 4 B/C/D proxy runs fit in about
38 GiB and checkpointing costs compute.

## Dataset Loading Contract

`src/train.py` expects training-schema CSVs:

```text
audio_path,text,duration,speaker_id,split,source_metadata
```

Gold governance CSVs use:

```text
audio_path,transcript,dataset_name,duration_sec,speaker_id,split,quality_score
```

Do not pass the governance schema directly to `src/train.py` unless the loader is
modified and tested.

During search and active Gold training:

```yaml
load_test_split: false
evaluate_test_after_training: false
```

The trainer uses train only for optimization and validation only for evaluation,
best-checkpoint selection, early stopping, and status reports.

## Active Full Gold Run

Config: `configs/full_training/gold_bcd_decoder_2e5.yaml`.

This run tests the best 30h proxy configuration on the full 207.115h Gold corpus.

| Setting | Value |
|---|---|
| Data root | `data/gold_master_training_schema/` |
| Train | 172,135 rows / 186.4037h |
| Validation | 6,068 rows / 10.3556h |
| Test | not loaded |
| Epochs | 1 |
| Steps | 5,380 |
| Batch / accumulation | 4 / 8 |
| Effective batch | 32 |
| BF16 | true |
| Gradient checkpointing | false |
| Duration bucketing | true |
| Dataloader workers | 8 |
| Eval/save | every 1000 steps |

Learning-rate schedule:

| Group | LR |
|---|---:|
| Encoder 0-7 | frozen |
| Encoder 8-15 | 2e-5 |
| Encoder 16-23 | 2e-5 |
| Encoder 24-31 | 2e-5 |
| Decoder + proj_out | 2e-5 |

Expected first validation: step 1000. Best model snapshots write to
`outputs_full_gold/best_model/` when validation WER improves.

## Best Completed Models

### Protected Partial FT Baseline

- Path: `models/partial_ft_usc_baseline/model/`
- Dataset: USC only
- Frozen: encoder 0-23
- Trainable: encoder 24-31 + decoder
- Epochs: 1
- Test WER: `0.2005258480`
- Test CER: `0.0529079419`

### USC Full FT Negative Result

- Metrics: `outputs_full_ft/test_metrics.json`
- All parameters trainable
- Encoder LR: `2e-6`
- Decoder LR: `8e-6`
- BF16
- Epochs: 1
- Test WER: `0.2221522737`
- Test CER: `0.0565825834`

Conclusion: full FT all layers on USC did not beat the partial baseline.

## LR-Search Results

All listed LR-search metrics are validation metrics on proxy data, not locked final
test metrics. Main proxy runs used `data/lr_search/main_30h/` with 26,249 train rows
and 847 validation rows. Most completed main proxy runs used 2 epochs, so
`ceil(26249 / 32) * 2 = 1642` optimizer steps.

### Best Proxy Runs

| Rank | Experiment | Mode | LR schedule | Best step | WER | CER |
|---:|---|---|---|---:|---:|---:|
| 1 | `phase4x_encoder_bcd_decoder_2e5_bs4_fast` | blockwise | A frozen; B/C/D/decoder `2e-5` | 1642 | 0.191341 | 0.048445 |
| 2 | `phase4x_encoder_b1e5_cd_decoder2e5_bs4_fast` | blockwise | A frozen; B `1e-5`; C/D/decoder `2e-5` | 1600 | 0.192737 | 0.048256 |
| 3 | `phase4x_main_encoder_cd_decoder_5em05_bs4_fast` | encoder 16-31 + decoder | encoder/decoder `5e-5` | 1642 | 0.193735 | 0.050009 |
| 4 | `phase4x_blockwise_c1e5_d5e5_decoder5e5_bs4_fast` | blockwise | A/B frozen; C `1e-5`; D/decoder `5e-5` | 1642 | 0.194334 | 0.068783 |
| 5 | `phase4x_main_encoder_cd_decoder_2em05_bs4_fast` | encoder 16-31 + decoder | encoder/decoder `2e-5` | 1642 | 0.194932 | 0.050900 |

The WER gap between ranks 1 and 2 is small, but rank 1 has better WER and was selected
for full Gold promotion.

### Important Negative Proxy Runs

| Experiment | Result | Lesson |
|---|---|---|
| `phase4x_full_encoder_decoder_2e5_bs1_safe` | WER 3.9112 at step 400 | full encoder at 2e-5 is unsafe/poor |
| `phase2_decoder_2em05` | WER 6.3134 | decoder-only at 2e-5 degraded badly |
| `phase1a_decoder_lr_5e5` | WER 2.6401 | decoder-only 5e-5 is too aggressive |
| `phase4x_main_all_blocks_aggressive_failed_bs2_20260627T054224Z` | failed quickly with no eval | all-block variants need caution |

## Resume and Safety

Resume command for the active run:

```bash
cd /home/mahmud/whisper-uz-ft
tmux new-session -d -s whisper_gold_ft \
  "export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   .venv/bin/python src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml --resume auto \
   2>&1 | tee -a outputs_full_gold/logs/full_gold_training.log"
```

Safety callbacks stop on non-finite loss, repeated unsafe grad norm, two consecutive
validation WER regressions, or substantial hallucination increase. Checkpoint integrity
is verified on save.

Do not resume from a checkpoint after changing tuning mode, optimizer groups, dataset,
scheduler, precision, or effective batch unless the run is intentionally restarted in a
new output directory.
