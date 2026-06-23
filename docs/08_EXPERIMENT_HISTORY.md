# Experiment History

## 1. Raw Whisper Baseline

Model:

- `openai/whisper-large-v3`.

Goal:

- Measure unadapted Uzbek ASR quality.

Observed results:

- Raw large-v3 on Uzbek full/reference evaluation: WER `1.0522`, CER `0.4590`.
- Mini validation baseline in `outputs/baseline_metrics.json`: WER `1.3799`, CER `0.8109`.

Key failure mode:

- Hallucinations.
- Turkish/Kazakh-like output.
- Wrong script and language prior.

Lesson:

- Whisper large-v3 has useful acoustic capability but harmful multilingual priors for Uzbek.
- Forced Uzbek decoding and fine-tuning are mandatory.

## 2. Mini Fine-Tune

Path:

- `outputs/mini/`

Config:

- `configs/mini_train.yaml`.

Result:

- `outputs/mini/test_metrics.json`
- WER: `0.4960674157`.
- CER: `0.1094431339`.
- Epoch: `2.0`.

Lesson:

- Even small adaptation sharply improves Uzbek.
- Dataset and normalization are central bottlenecks.

## 3. Partial FT USC Baseline

Archived path:

- `archive/partial_ft_usc/`

Model path:

- `archive/partial_ft_usc/model/`

Training strategy:

- Base: `openai/whisper-large-v3`.
- Dataset: USC clean corpus only.
- Freeze encoder blocks 0-23.
- Train encoder blocks 24-31.
- Train full decoder.
- Trainable parameters: `1,063,930,880`.
- Frozen parameters: `479,559,680`.
- Precision: FP16.
- Epochs: `1`.
- Effective batch size: `32`.
- LR: `1e-5`.

Result:

- Test WER: `0.2005258480`.
- Test CER: `0.0529079419`.
- Test loss: `0.2275837064`.

Lesson:

- Partial FT creates a strong baseline.
- However, frozen lower encoder layers may preserve multilingual priors that are suboptimal for Uzbek-only quality.

## 4. Full FT Dry Run

Path:

- `outputs_full_ft_dry_run/`.

Config:

- `configs/full_ft_dry_run.yaml`.

Purpose:

- Verify full FT BF16 stability before multi-hour training.

Result:

- 100-step dry run completed.
- Loss dropped from about `28.31` to `5.269`.
- Peak VRAM about `30 GiB`.
- Final mini test WER: `0.5567`.
- Final mini test CER: `0.1357`.

Lesson:

- Full FT BF16 is feasible on A40 48GB with gradient checkpointing.

## 5. Full FT USC Current Run

Path:

- Output: `outputs_full_ft/`.
- Logs: `logs/full_ft_uzbek.log`.

Initial plan:

- Full FT for 4 epochs.
- BF16.
- Encoder LR `2e-6`.
- Decoder LR `8e-6`.
- Eval/save every 1000 steps.

Current user decision:

- Run only 1 epoch.

Current mitigation:

- `configs/full_ft_uzbek.yaml` patched to `epochs: 1`.
- `scripts/guard_one_epoch_resume.sh` waited for checkpoint 1000, stopped the old process, and restarted with the one-epoch config.

Latest reliable milestone:

- Step 1000.
- Loss `11.7892`.
- Epoch `0.3212`.
- Eval WER `0.3332`.
- Eval CER `0.09192`.
- Checkpoint `outputs_full_ft/checkpoint-1000` exists.
- Resume failed because Transformers `5.12.1` requires PyTorch `>=2.6` to load `optimizer.pt`; local PyTorch is `2.5.1+cu121`.

Lesson so far:

- Training is stable enough to reach step 1000.
- Evaluation overhead is significant.
- Runtime control must be aligned with user-specified epoch count before launch.
- Dependency compatibility must be verified for checkpoint resume, not only initial training.

## 6. Gold Corpus Creation

Goal:

- Build a larger Gold corpus from high-trust Uzbek datasets.

Included:

- USC.
- Common Voice Uzbek cleaned mirror.
- FLEURS Uzbek.

Blocked:

- FeruzaSpeech gated manual HF access.

Result:

- Raw available Gold: 184,325 rows, `207.27h`.
- Final Gold master: 184,140 rows, `207.12h`.
- Train: `186.40h`.
- Val: `10.36h`.
- Test: `10.36h`.
- Split validation passed with 0 missing paths, 0 exact hash leakage, and 0 known speaker leakage.

Lesson:

- The project now has enough Gold data for a stronger next experiment after the current USC-only full FT is evaluated.
