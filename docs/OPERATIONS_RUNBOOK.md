# Operations Runbook

Last cleaned for obsolete documentation sections: `2026-07-01T05:07:45Z`.

## Resume Stage 1 After a Valid Checkpoint

```bash
cd /home/mahmud/whisper-uz-ft
export PYTHONPATH=/home/mahmud/whisper-uz-ft/src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
.venv/bin/python src/train.py --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml --resume auto
```

Only resume from a checkpoint that passes `verify_checkpoint` in `src/train.py`. If no valid checkpoint exists, restart intentionally; do not claim exact resume from a standalone best-model snapshot.

## Disk Safety

- Keep `/home/mahmud/.cache/huggingface/datasets` small or absent during large training.
- The no-cache trainer should not create persistent Whisper feature Arrow files.
- Stage 1 checkpoint retention is `save_total_limit: 2`.
- Do not delete `models/partial_ft_usc_baseline/` or `outputs_full_gold/best_model/`.

## Training Sanity Check

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=/home/mahmud/whisper-uz-ft/src
.venv/bin/python src/train.py   --config configs/stage1/gold_silver_bcd_decoder_2e5_nocache.yaml   --sanity-check   --sanity-report outputs_stage1_gold_silver_nocache/metrics/sanity_report.json
```
