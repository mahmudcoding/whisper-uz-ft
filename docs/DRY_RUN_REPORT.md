# Full Fine-Tuning Dry Run Report

Generated: 2026-06-23 UTC

## Command

```bash
cd /home/mahmud/whisper-uz-ft
. .venv/bin/activate
PYTHONPATH=src python src/train.py --config configs/full_ft_dry_run.yaml 2>&1 | tee logs/full_ft_dry_run.log
```

## Config

- Base model: `openai/whisper-large-v3`
- Split prefix: `mini`
- Full FT: all layers trainable
- Trainable params: 1,543,490,560
- Precision: BF16
- CUDA: yes
- Max steps: 100
- Batch: 1
- Grad accumulation: 8
- LR: `8e-6`
- Warmup ratio: `0.1`
- Weight decay: `0.03`
- SpecAugment: enabled
- Eval/save steps: 50

## Results

Training completed successfully with exit code 0.

Loss logs:

- Step 10: loss `28.31`, grad norm `158.2`
- Step 20: loss `16.17`, grad norm `100`
- Step 30: loss `13.13`, grad norm `92.23`
- Step 40: loss `11.38`, grad norm `224.1`
- Step 50: loss `12.13`, grad norm `124.2`
- Step 60: loss `9.771`, grad norm `94.92`
- Step 70: loss `7.696`, grad norm `91.71`
- Step 80: loss `6.122`, grad norm `83.99`
- Step 90: loss `5.441`, grad norm `102.2`
- Step 100: loss `5.269`, grad norm `103.9`

Validation:

- Step 50: eval loss `1.431`, WER `0.6139`, CER `0.1570`, runtime `314.2s`
- Step 100: eval loss `0.8193`, WER `0.5792`, CER `0.1716`, runtime `330.0s`

Final mini test:

- Test loss: `0.7518193126`
- Test WER: `0.5567415730`
- Test CER: `0.1356780327`
- Runtime: `288.4579s`

Resource usage:

- Peak allocated VRAM from dry report: `30017.736 MiB`
- Checkpoint size: about 18 GiB each
- Dry-run output directory: 41 GiB

Artifacts:

- Log: `logs/full_ft_dry_run.log`
- Checkpoints: `outputs_full_ft_dry_run/checkpoint-50`, `outputs_full_ft_dry_run/checkpoint-100`
- Final model: `outputs_full_ft_dry_run/final_model`
- Test metrics: `outputs_full_ft_dry_run/test_metrics.json`

## Validation Checklist

- BF16 enabled: passed
- Full FT enabled: passed, 100% trainable parameters
- CUDA path: passed
- 100 training steps: passed
- Loss decreases: passed
- NaN detection: no NaNs observed
- Gradient explosion: no unsafe grad norm; max logged grad norm `224.1`
- VRAM stability: passed, peak about 30.0 GiB
- Checkpoint save: passed at steps 50 and 100 with `CHECKPOINT_OK`
- Eval works: passed
- Final test evaluation works: passed

## Warnings

- Transformers warned about duplicate suppress-token logits processors during generation. This did not fail the run.
- Transformers warned about BPE cleanup; current output path is still valid, but tokenizer cleanup should remain disabled in future evaluation if exposed.
- Early stopping warned after the final `test` evaluation because the callback expected `eval_wer`, not `test_wer`. Train-time validation produced `eval_wer`, so this is not a blocker for model selection.
- The dry-run printed `dry_report` but did not persist it because `dry_run_report` was not configured. Future dry-run configs should set this path.

## Decision Gate

Recommendation: A, launch training as-is only after user approval.

Evidence: BF16 full FT is stable on A40, loss drops strongly, checkpointing works, evaluation works, and VRAM has about 18 GiB headroom. The main caveat is runtime: current eval cadence makes the 4-epoch run likely take about 74-80 hours.

