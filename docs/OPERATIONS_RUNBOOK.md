# Operations Runbook

Last rebuilt: `2026-07-01T04:50:03Z`.

## Check Live State

```bash
cd /home/mahmud/whisper-uz-ft
tmux ls
pgrep -af 'src/train.py|run_experiment.py|score_teacher.py|launch_stage1'
nvidia-smi
df -h /home/mahmud /home/mahmud/whisper-uz-ft
tail -n 40 outputs_stage1_gold_silver_nocache/logs/training_metrics.jsonl
```

## Monitor Active Stage 1

```bash
tmux attach -t whisper_stage1_gold_silver_nocache
tail -f outputs_stage1_gold_silver_nocache/logs/stage1_gold_silver_nocache_training.log
tail -f outputs_stage1_gold_silver_nocache/logs/stage1_gold_silver_nocache_system.log
```

## Resume Stage 1 After Valid Checkpoint

```bash
cd /home/mahmud/whisper-uz-ft
export PYTHONPATH=/home/mahmud/whisper-uz-ft/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
.venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml --resume auto
```

Only resume from a checkpoint that passes `verify_checkpoint` in `src/train.py`. If no checkpoint exists, restarting from scratch is cleaner than pretending to resume.

## Disk Safety

- Keep `/home/mahmud/.cache/huggingface/datasets` small or absent during long training.
- The active no-cache trainer should not create persistent Whisper feature Arrow files.
- Checkpoint retention for Stage 1 is `save_total_limit: 2`.
- Do not delete `models/partial_ft_usc_baseline/` or `outputs_full_gold/best_model/`.

## Sanity Check Training Pipeline

```bash
.venv/bin/python src/train.py   --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml   --sanity-check   --sanity-report outputs_stage1_gold_silver_nocache/metrics/sanity_report.json
```
