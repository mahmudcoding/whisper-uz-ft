# Training Pipeline

## Main Files

- `src/train.py`
- `src/model.py`
- `configs/train.yaml`
- `configs/full_ft_uzbek.yaml`
- `configs/full_ft_dry_run.yaml`
- `configs/mini_train.yaml`
- `scripts/start_full_training.sh`
- `scripts/guard_one_epoch_resume.sh`
- `scripts/system_monitor.sh`

## Model Loading

`src/model.py` loads `openai/whisper-large-v3` with:

- `language="uz"`.
- `task="transcribe"`.
- Forced decoder IDs from Whisper processor.
- `suppress_tokens=[]`.
- `use_cache=False`.
- Optional gradient checkpointing.

The loader supports:

- Partial FT: freeze encoder except last N blocks and train full decoder.
- Full FT: all parameters trainable when `train_last_encoder_blocks: all`.

## Current Full FT Config

File:

```bash
configs/full_ft_uzbek.yaml
```

Current intended training plan:

- Model: `openai/whisper-large-v3`.
- Dataset: USC splits in `data/`.
- Full FT: `train_last_encoder_blocks: all`.
- Trainable parameters: all `1,543,490,560`.
- Epochs: `1`.
- Precision: BF16.
- Batch size per device: `1`.
- Gradient accumulation: `32`.
- Effective batch: `32`.
- Encoder LR: `2e-6`.
- Decoder LR: `8e-6`.
- Weight decay: `0.03`.
- Warmup ratio: `0.1`.
- Scheduler: cosine.
- Gradient clipping: `max_grad_norm: 1.0`.
- Eval steps: `1000`.
- Save steps: `1000`.
- Early stopping patience: `5`.
- Metric for best model: `wer`.
- Greater is better: `false`.
- SpecAugment enabled.

## Layer-Wise Learning Rate

Implemented in `build_layerwise_optimizer()` in `src/train.py`.

Parameter groups:

- Encoder params use `encoder_learning_rate`.
- Decoder and `proj_out` params use `decoder_learning_rate`.
- Bias and layer norm parameters use zero weight decay.
- Optimizer: PyTorch `AdamW`.

Rationale:

- Uzbek failures appear dominated by decoder language-prior errors and hallucinations.
- The encoder likely retains useful acoustic features.
- Decoder should adapt more aggressively than encoder.

## Safety Controls

`SafetyCallback` in `src/train.py`:

- Stops on non-finite loss.
- Tracks unsafe gradient norms.
- Saves before stopping where possible.
- Verifies checkpoint structure after save.

`ProductionStatusCallback`:

- Writes milestone reports at configured steps.
- Writes eval reports.
- Captures GPU snapshots.
- Generates sample predictions during eval.
- Tracks hallucination and language-confusion indicators.
- Can stop on two consecutive WER regressions.
- Can stop on substantial hallucination-rate increase.

## Resume Behavior

`src/train.py` supports:

```bash
python src/train.py --config configs/full_ft_uzbek.yaml --resume auto
```

`--resume auto` finds the latest checkpoint in the output directory.

Checkpoint verification checks:

- `trainer_state.json`
- `training_args.bin`
- model weights file
- parseable trainer state

Important:

- Optimizer and scheduler state are preserved when resuming from a complete Trainer checkpoint.
- The one-epoch guard waits for `trainer_state.json`, `optimizer.pt`, and `scheduler.pt` in `checkpoint-1000`.

## Active One-Epoch Guard

Because the full FT process was launched with a 4-epoch config before the user changed the requirement, `scripts/guard_one_epoch_resume.sh` was added.

Behavior:

1. Wait for a complete `outputs_full_ft/checkpoint-1000`.
2. Stop the old training pane.
3. Respawn the tmux training pane with the patched one-epoch config.
4. Resume automatically from the latest checkpoint.

The script has completed once and exposed a resume blocker: local PyTorch `2.5.1+cu121` cannot load Trainer optimizer/scheduler `.pt` files under Transformers `5.12.1`, which requires PyTorch `>=2.6` due to CVE-2025-32434 safety checks.

Resolution options:

1. Preferred: upgrade PyTorch to `>=2.6` in the virtual environment, then resume from `outputs_full_ft/checkpoint-1000`.
2. Alternative: start a fresh one-epoch run from model weights only, accepting loss of optimizer/scheduler state. This is not equivalent to preserving the step-1000 state.
3. Avoid undocumented monkeypatches around `torch.load` safety checks.

## Running Training

Manual launch pattern:

```bash
cd /home/mahmud/whisper-uz-ft
tmux new-session -d -s whisper_full_ft_uzbek -n train \
  "source .venv/bin/activate; export PYTHONPATH=src PYTHONUNBUFFERED=1; python src/train.py --config configs/full_ft_uzbek.yaml --resume auto 2>&1 | tee -a logs/full_ft_uzbek.log"
tmux new-window -t whisper_full_ft_uzbek -n monitor \
  "bash scripts/system_monitor.sh logs/full_ft_uzbek_system.log"
```

Prefer existing launch scripts if they match the intended config.

## Stop Conditions

Stop or investigate immediately if:

- NaN or infinite loss appears.
- Unsafe gradient norm repeats.
- CUDA OOM occurs.
- Eval WER worsens for two consecutive evals.
- Hallucination indicators increase substantially.
- Training stalls with no log updates and no GPU activity.
- Resume fails on optimizer/scheduler load due PyTorch version safety restrictions.
