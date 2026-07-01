# Current Project Status

Last rebuilt from repository reality: `2026-07-01T04:51:05Z`.

## Live Runtime

- Active training: `whisper_stage1_gold_silver_nocache` tmux session is running Stage 1 Gold+Silver fine-tuning from scratch.
- Command source: `/tmp/launch_stage1_nocache.sh`.
- Config: `configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml`.
- Output root: `outputs_stage1_gold_silver_nocache/`.
- Current logged optimizer step at rebuild time: `100` / `21339` (0.469%).
- Latest train log row: `{'step': 100, 'loss': 13.520646667480468, 'grad_norm': 79.82109832763672, 'learning_rate': 9.278350515463919e-07, 'epoch': 0.004686310116571964}`.
- ETA from early step rate: `~27.1h` training time excluding evaluations/checkpoint writes, based on ~4.6 sec/optimizer step early run rate.
- GPU snapshot: `NVIDIA A40, 100, 38777, 46068, 289.02, 65`.

Tmux sessions:

```text
codex: 1 windows (created Mon Jun 22 16:25:30 2026) (attached)
whisper_stage1_gold_silver_nocache: 1 windows (created Wed Jul  1 04:42:13 2026)
```

Active Stage 1 process tree:

```text
1950321 1950316 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952142 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952143 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952144 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952145 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952146 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952147 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952148 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
1952149 1950321 .venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml
```

## Current Training Configuration

- Base model: `openai/whisper-large-v3`.
- Objective: Uzbek-only ASR; forced `language=uz`, `task=transcribe`.
- Dataset: `data/gold_silver_training/` train split, with gold-only validation; test is not loaded during training.
- Trainable blocks: encoder B/C/D layers 8-31 plus decoder.
- Frozen blocks: encoder A layers 0-7.
- LR: encoder B/C/D `2e-5`, decoder `2e-5`.
- Precision: BF16.
- Batch: per-device batch 4, gradient accumulation 8, effective batch 32.
- Scheduler: cosine with `warmup_ratio: 0.10`.
- Checkpoints: every 1000 steps, `save_total_limit: 2`, best model saved separately by validation WER.
- Dataset caching: Hugging Face datasets cache disabled; large persistent Whisper feature cache removed. Features are computed on-the-fly by `OnTheFlySpeechDataset` in `src/train.py`.

## Best Models

- Best protected baseline: `models/partial_ft_usc_baseline/model/`, partial FT on USC, test WER `20.05%`, CER `5.29%`.
- Best finished Gold-only model: `outputs_full_gold/best_model/`, validation WER `14.50%`, CER `3.67%` at step `5000`.
- Current Stage 1 Gold+Silver has no validation result yet after restart; previous failed cached run reached step 1000 with WER 29.42%, then was deleted.

## Disk and Cache

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda1       2.0T  323G  1.7T  17% /
/dev/vda1       2.0T  323G  1.7T  17% /
```

```text
5.8G	/home/mahmud/.cache/huggingface
23G	/home/mahmud/whisper-uz-ft
139G	/home/mahmud/datasets
```

## Git State

The worktree is intentionally dirty. Important current changes include the no-cache training patch, Stage 1 no-cache config, and rebuilt documentation. Do not revert unrelated existing changes.

```text
M AGENTS.md
 M PROJECT_CONTEXT_EXPORT.txt
 M configs/silver_datasets.yaml
 M docs/AGENT_BRIEF.md
 M docs/DATA_GOVERNANCE.md
 M docs/DECISION_LOG.md
 M docs/EXPERIMENT_LEDGER.md
 M docs/FAILURE_LOG.md
 M docs/MODEL_REGISTRY.md
 M docs/OPERATIONS_RUNBOOK.md
 M docs/README.md
 M docs/STATUS.md
 M docs/TRAINING_AND_SEARCH.md
 M docs/state.json
 M outputs_full_gold/checkpoints/runs/Jun28_16-30-32_vm-do01lr81-0/events.out.tfevents.1782664232.vm-do01lr81-0.264671.0
 D outputs_lr_search/phase4x_main_all_blocks_aggressive/checkpoint-200/config.json
 D outputs_lr_search/phase4x_main_all_blocks_aggressive/checkpoint-200/generation_config.json
 D outputs_lr_search/phase4x_main_all_blocks_aggressive/checkpoint-200/preprocessor_config.json
 D outputs_lr_search/phase4x_main_all_blocks_aggressive/checkpoint-200/trainer_state.json
 M scripts/silver_pipeline/finalize_silver.py
 M src/train.py
?? configs/stage1/
?? data/
?? outputs_full_gold/best_model/
?? outputs_full_gold/checkpoints/run_metrics.json
?? outputs_full_gold/metrics/best_metrics.json
?? outputs_full_gold/metrics/dry_run_report.json
?? outputs_full_gold/metrics/validation_metrics_history.jsonl
?? outputs_full_gold/status_reports/status_step_1000_eval.json
?? outputs_full_gold/status_reports/status_step_1000_eval.md
?? outputs_full_gold/status_reports/status_step_1000_milestone.json
?? outputs_full_gold/status_reports/status_step_1000_milestone.md
?? outputs_full_gold/status_reports/status_step_2000_eval.json
?? outputs_full_gold/status_reports/status_step_2000_eval.md
?? outputs_full_gold/status_reports/status_step_2000_milestone.json
?? outputs_full_gold/status_reports/status_step_2000_milestone.md
?? outputs_full_gold/status_reports/status_step_3000_eval.json
?? outputs_full_gold/status_reports/status_step_3000_eval.md
?? outputs_full_gold/status_reports/status_step_3000_milestone.json
?? outputs_full_gold/status_reports/status_step_3000_milestone.md
?? outputs_full_gold/status_reports/status_step_4000_eval.json
?? outputs_full_gold/status_reports/status_step_4000_eval.md
?? outputs_full_gold/status_reports/status_step_4000_milestone.json
?? outputs_full_gold/status_reports/status_step_4000_milestone.md
?? outputs_full_gold/status_reports/status_step_5000_eval.json
?? outputs_full_gold/status_reports/status_step_5000_eval.md
?? outputs_full_gold/status_reports/status_step_5000_milestone.json
?? outputs_full_gold/status_reports/status_step_5000_milestone.md
?? outputs_full_gold/status_reports/status_step_500_milestone.json
?? outputs_full_gold/status_reports/status_step_500_milestone.md
?? outputs_full_gold/status_reports/status_step_5380_eval.json
?? outputs_full_gold/status_reports/status_step_5380_eval.md
?? outputs_stage1_gold_silver_nocache/
?? reports/silver_dedup_report/summary.json
?? reports/silver_quality_report/SILVER_CORPUS_REPORT.md
?? reports/silver_quality_report/summary.json
?? reports/silver_quality_report/teacher_scoring_config.json
?? scripts/silver_pipeline/score_teacher_parallel.py
```
