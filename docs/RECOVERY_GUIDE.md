# Recovery Guide

Generated: 2026-06-23 UTC

This guide assumes the machine is available and only the documentation plus project source need to be reconstructed.

## 1. Environment

```bash
cd /home/mahmud
python3 -m venv whisper-uz-ft/.venv
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r setup/requirements.lock.txt
```

Verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 2. Dataset

Download or recreate USC manifests so these files exist:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Expected totals:

- train: 99,617 rows, 96.1401h
- val: 3,762 rows, 3.9967h
- test: 3,821 rows, 4.4930h

Validate audio:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/validate_audio.py
```

## 3. Training

Use the safe full training config:

```bash
cat configs/train.yaml
```

Expected important settings:

- epochs: 1
- per_device_batch_size: 2
- gradient_accumulation_steps: 16
- learning_rate: 1e-5
- fp16: true
- gradient_checkpointing: true
- eval_steps: 500
- save_steps: 500
- train_last_encoder_blocks: 8

Launch:

```bash
cd /home/mahmud/whisper-uz-ft
bash scripts/start_full_training.sh
```

Resume:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
PYTHONPATH=src python src/train.py --config configs/train.yaml --resume auto
```

Monitor:

```bash
tmux attach -t whisper_full_training
tail -f logs/full_training.log
tail -f logs/full_training_system.log
nvidia-smi
```

## 4. Evaluation

Evaluate the final model:

```bash
source .venv/bin/activate
PYTHONPATH=src python benchmark/eval_suite.py \
  --model-path outputs/final_model \
  --manifest data/test.csv \
  --output reports/eval_final_model_test.json \
  --batch-size 2 \
  --precision fp16
```

Expected current full-run metrics:

- WER: 0.2005
- CER: 0.0529

## 5. Benchmarking

Offline faster-whisper benchmark example:

```bash
bash benchmark/scripts/run_benchmark.sh \
  --engine faster-whisper \
  --model-path large-v3 \
  --dataset smoke \
  --precision fp16 \
  --batch-size 2 \
  --beam-size 1 \
  --mode offline
```

Capacity planning:

```bash
python benchmark/scripts/capacity_planner.py \
  --results benchmark/results \
  --hardware-config benchmark/configs/hardware_costs.yaml \
  --output benchmark/reports/capacity_plan.json
```

## 6. Normalization and Filtering

Run tests:

```bash
PYTHONPATH=src python -m text_normalization.tests
```

Score a manifest:

```bash
PYTHONPATH=src python src/filtering/filter_dataset.py \
  --input-csv data/train.csv \
  --output-csv reports/train_quality_scores.csv \
  --bad-csv reports/train_bad_samples.csv
```

## 7. Troubleshooting

CUDA OOM:

- Reduce `per_device_batch_size`.
- Keep `gradient_checkpointing: true`.
- Check for stray processes with `nvidia-smi`.

Resume failure:

- Verify checkpoint has `trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `scaler.pt`, and `model.safetensors`.
- Resume with `--resume auto`.

Bad WER:

- Confirm language/task settings are Uzbek/transcribe.
- Evaluate normalized and raw WER.
- Inspect text normalization and label script consistency.

Slow evaluation:

- Use `generation_num_beams: 1` for speed baselines.
- Increase batch only after checking VRAM.

## Uzbek-Only Full Fine-Tuning Recovery

Validate full fine-tuning setup:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
PYTHONPATH=src python src/train.py \
  --config configs/full_ft_uzbek.yaml \
  --sanity-check \
  --sanity-report logs/full_ft_sanity_report.json
```

Before a full run, create a copy of `configs/full_ft_uzbek.yaml` with `max_steps: 200` and verify no OOM/NaN. Then launch the 4-epoch config only if stable.

All inference paths should force:

- `language=uz`
- `task=transcribe`
