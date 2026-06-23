# Current State

Last audited: 2026-06-23 UTC.

## What Is Happening Right Now

The Uzbek-only full fine-tuning run is currently stopped after a resume failure.

- tmux session: `whisper_full_ft_uzbek`
- Train pane: `whisper_full_ft_uzbek:train`
- Monitor pane: `whisper_full_ft_uzbek:monitor`
- Main training command: `python src/train.py --config configs/full_ft_uzbek.yaml --resume auto`
- Training log: `logs/full_ft_uzbek.log`
- System monitor log: `logs/full_ft_uzbek_system.log`
- Status reports: `logs/full_ft_status_reports/`

The run was originally launched with `epochs: 4`, but the user changed the requirement to one epoch. The config was patched to:

```yaml
epochs: 1
```

The one-epoch guard completed its intended control action:

- Script: `scripts/guard_one_epoch_resume.sh`
- Guard log: `logs/full_ft_one_epoch_guard.log`
- It waited for complete `outputs_full_ft/checkpoint-1000`.
- It stopped the old 4-epoch process.
- It restarted the train pane with the patched one-epoch config.
- The restart failed while loading optimizer/scheduler state.

Failure:

```text
ValueError: Due to a serious vulnerability issue in torch.load, even with weights_only=True,
Transformers requires users to upgrade torch to at least v2.6 in order to load optimizer.pt.
```

Current local PyTorch is `2.5.1+cu121`; Transformers is `5.12.1`. Resume from the complete Trainer checkpoint requires either upgrading PyTorch to `>=2.6` or implementing a deliberate, documented workaround. Do not restart blindly without resolving this.

## Latest Observed Training State

Latest reliable milestone/eval:

- File: `logs/full_ft_status_reports/status_step_1000_milestone.json`
- Step: `1000 / 12456` from the original 4-epoch plan.
- Reported progress under old plan: `8.03%`.
- Latest train loss at milestone: `11.7892`.
- Grad norm: `145.89`.
- LR: `1.6035e-06`.
- Epoch: `0.3212`.
- VRAM used: about `29.1 GiB / 45.0 GiB`.
- A40 temperature: about `64C`.
- Eval WER at step 1000: `0.3332`.
- Eval CER at step 1000: `0.09192`.
- Eval hallucination rate: `0.0002658`.
- Eval language-confusion rate: `0.0002658`.
- Checkpoint: `outputs_full_ft/checkpoint-1000` exists and contains `trainer_state.json`, `optimizer.pt`, and `scheduler.pt`.

## Current Best Model

Best completed model:

- Registry ID: `partial_ft_usc_baseline`
- Path: `archive/partial_ft_usc/model/`
- Source copy: `outputs/final_model/`
- Training type: partial fine-tune on USC.
- Test WER: `0.2005258480` (`20.05%`).
- Test CER: `0.0529079419` (`5.29%`).

This archived baseline is sacred. Do not modify it.

## Active Datasets

USC-only training data currently used by active full FT:

- `data/train.csv`: 99,617 rows.
- `data/val.csv`: 3,762 rows.
- `data/test.csv`: 3,821 rows.
- Total clean USC: about `104.63h`.

Gold master corpus created but not yet used for the active run:

- Path: `data/gold_master/`
- Total: 184,140 rows, `207.12h`.
- Train: 172,135 rows, `186.40h`.
- Val: 6,068 rows, `10.36h`.
- Test: 5,937 rows, `10.36h`.
- Missing audio paths: 0.
- Exact content hash leakage across splits: 0.
- Known speaker leakage across splits: 0.

## Current Bottlenecks

P0 bottlenecks:

- Current full FT must finish one epoch and be evaluated against the partial FT baseline.
- FeruzaSpeech is blocked by gated Hugging Face access.
- Silver datasets have not yet been acquired.

P1 bottlenecks:

- Quality scoring is heuristic; full teacher-ASR agreement scoring has not been applied to the full Gold corpus.
- FLEURS does not expose reliable speaker IDs, so speaker leakage cannot be fully excluded inside FLEURS.
- Current active training still uses USC-only splits, not the new Gold master corpus.

## Next Milestone

1. Resolve PyTorch/Transformers resume incompatibility.
2. Resume from `outputs_full_ft/checkpoint-1000` with `epochs: 1`.
3. Finish one epoch of full FT on USC.
4. Evaluate the resulting model on USC test and language-confusion benchmarks.
5. Compare against `partial_ft_usc_baseline`.
6. Decide whether to train the next run on `data/gold_master/` with weighted sampling.

## Monitoring Commands

```bash
tmux attach -t whisper_full_ft_uzbek
tail -f logs/full_ft_uzbek.log
tail -f logs/full_ft_one_epoch_guard.log
tail -f logs/full_ft_uzbek_system.log
nvidia-smi
```

## Do Not Do

- Do not start Silver dataset work before evaluating the one-epoch USC full FT.
- Do not overwrite `archive/partial_ft_usc/`.
- Do not relaunch with `--resume auto` until the PyTorch `>=2.6` resume issue is resolved.
- Do not trust archived docs as current truth without cross-checking these numbered docs.
