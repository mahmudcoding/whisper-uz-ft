# AI Agent Context

This file is the compressed memory injection for new AI coding agents.

## Identity

Project: Uzbek Whisper ASR fine-tuning and production benchmarking.

Root:

```bash
/home/mahmud/whisper-uz-ft
```

Goal:

- Build the best open-weight Uzbek ASR model using `openai/whisper-large-v3`.
- Optimize only Uzbek WER/CER.
- Catastrophic forgetting of non-Uzbek languages is acceptable.

## Current State

Full FT is currently stopped after a resume failure:

- Session: `whisper_full_ft_uzbek`.
- Train log: `logs/full_ft_uzbek.log`.
- Monitor log: `logs/full_ft_uzbek_system.log`.
- Config: `configs/full_ft_uzbek.yaml`.
- Output: `outputs_full_ft/`.

Important: the run was launched when config had `epochs: 4`, but user changed to one epoch. Config is now patched to `epochs: 1`. Guard script `scripts/guard_one_epoch_resume.sh` completed its checkpoint wait and restart attempt.

Current blocker:

- `outputs_full_ft/checkpoint-1000` exists.
- Resume failed because Transformers `5.12.1` requires PyTorch `>=2.6` to load `optimizer.pt`; local PyTorch is `2.5.1+cu121`.
- Do not start another full training run until this is resolved.

## Best Completed Model

Registry ID: `partial_ft_usc_baseline`.

- Path: `archive/partial_ft_usc/model/`.
- Source: `outputs/final_model/`.
- Dataset: USC only.
- Partial FT: frozen encoder blocks 0-23, trained blocks 24-31 plus decoder.
- WER: `20.05%`.
- CER: `5.29%`.

Never modify `archive/partial_ft_usc/`.

## Active Training Config

Full Uzbek-only FT:

- All `1,543,490,560` params trainable.
- BF16.
- Encoder LR `2e-6`.
- Decoder LR `8e-6`.
- Epochs `1`.
- Effective batch `32`.
- Gradient checkpointing enabled.
- Gradient clipping `1.0`.
- Eval/save every 1000 steps.
- Forced Uzbek decoding.
- Early stopping configured.
- SpecAugment enabled.

## Datasets

Current active run uses USC:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `104.63h`.

Gold master built but not yet used:

- `data/gold_master/`
- `207.12h`.
- Includes USC, Common Voice Uzbek, FLEURS Uzbek.
- FeruzaSpeech blocked by gated HF access.

Gold master schema differs from current training schema; verify before training on it.

## Critical Decisions

- Uzbek-only quality is the only optimization target.
- Force `language="uz"` and `task="transcribe"`.
- Full FT is being tested because partial FT may preserve harmful multilingual priors.
- Layer-wise LR is preferred over uniform LR.
- Gold data must be normalized, deduped, and quality-scored.
- Silver/Bronze data must not be used raw.

## What Not To Do

- Do not overwrite or delete archived partial FT baseline.
- Do not trust old docs in `docs/archive/` as current truth.
- Do not train on Silver/Bronze before current full FT is evaluated.
- Do not launch 4 epochs unless the user explicitly approves.
- Do not allow automatic language detection in Uzbek ASR inference/evaluation.
- Do not remove the one-epoch guard until its job is complete.

## Immediate Next Tasks

1. Check current run:

```bash
tmux has-session -t whisper_full_ft_uzbek
tail -f logs/full_ft_uzbek.log
tail -f logs/full_ft_one_epoch_guard.log
```

2. Resolve PyTorch/Transformers resume incompatibility.

3. Resume from `outputs_full_ft/checkpoint-1000` with `epochs: 1`.

4. After training finishes, evaluate full FT against `partial_ft_usc_baseline`.

5. Update:

- `01_CURRENT_STATE.md`
- `08_EXPERIMENT_HISTORY.md`
- `10_MODEL_REGISTRY.md`
- `12_FAILURES_AND_LESSONS.md` if something fails.

## Documentation Rule

Before modifying code/config/data/training state, read:

- `01_CURRENT_STATE.md`
- `11_DECISIONS_AND_RATIONALE.md`
- `15_AI_AGENT_CONTEXT.md`
- `DOCUMENTATION_POLICY.md`
