# Agent Brief

This is the compressed handoff for a new AI coding session. Read this plus
`AGENTS.md` before acting.

## Mission

Build the best open-weight Uzbek ASR model using `openai/whisper-large-v3`. Optimize
Uzbek WER/CER only. Multilingual forgetting is acceptable.

## Current Runtime

As of 2026-06-28T16:56:49Z, full Gold fine-tuning is running.

```bash
tmux attach -t whisper_gold_ft
tail -f /home/mahmud/whisper-uz-ft/outputs_full_gold/logs/full_gold_training.log
tail -f /home/mahmud/whisper-uz-ft/outputs_full_gold/logs/training_metrics.jsonl
nvidia-smi
```

Active command:

```bash
.venv/bin/python src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml --resume auto
```

Latest inspected progress: about step `345 / 5380`, latest metric step `340`, no first
validation yet, about 38 GiB VRAM used on A40. First validation/checkpoint should
happen at step 1000.

## Current Best Artifacts

Best completed locked-test model:

- `models/partial_ft_usc_baseline/model/`;
- WER `0.2005258480`;
- CER `0.0529079419`;
- immutable protected baseline.

Best proxy LR-search configuration:

- `phase4x_encoder_bcd_decoder_2e5_bs4_fast`;
- main 30h proxy validation WER `0.1913407821`;
- CER `0.0484449599`;
- trainable: encoder 8-31 + decoder;
- frozen: encoder 0-7;
- LR: B/C/D/decoder `2e-5`;
- batch 4, grad accumulation 8, BF16, no gradient checkpointing.

The active full Gold run is testing this proxy winner on the full Gold corpus. It is
not yet a promoted model.

## Core Invariants

- Never modify or delete `models/partial_ft_usc_baseline/`.
- Never use test data for training, LR search, early stopping, checkpoint selection, or
  model ranking.
- Force Whisper decoding with `language="uz"` and `task="transcribe"`.
- Do not train raw Silver/Bronze.
- Do not restart or duplicate a training job without inspecting tmux, PIDs, logs,
  output directory, and checkpoints.
- Do not use an in-project fine-tuned model as Silver teacher; use Kotib or another
  independent teacher.
- Do not assume docs are fresher than live artifacts; reconcile and update docs.

## Data State

Gold:

- `data/gold_master/`;
- `data/gold_master_training_schema/`;
- USC + Common Voice Uzbek + FLEURS Uzbek;
- total 184,140 rows / 207.1150h;
- train 172,135 rows / 186.4037h;
- validation 6,068 rows / 10.3556h;
- test 5,937 rows / 10.3557h;
- no known path/hash/speaker leakage.

Silver:

- `data/silver_master/`;
- finalized FeruzaSpeech only, train-only, 12,854 rows / 57.8279h;
- large Silver exports exist but are not finalized;
- teacher scoring stopped at 129,007 / 535,728 candidates.

FeruzaSpeech was moved from Gold to Silver on 2026-06-27 because of restrictive gated
terms.

## Training Stack

- Trainer: `src/train.py`.
- Model freezing: `src/model.py`.
- Optimizer: custom AdamW groups.
- Scheduler: HuggingFace cosine scheduler from config.
- Active warmup: `warmup_ratio: 0.10`.
- Metrics: validation WER/CER, hallucination rate, language-confusion rate.
- Best model snapshot: `BestModelSnapshotCallback` writes to configured
  `best_model_dir` when validation WER improves.
- Status reports: `outputs_full_gold/status_reports/`.

## Important Results

Locked test results:

| Model | Data | Strategy | WER | CER | Conclusion |
|---|---|---|---:|---:|---|
| Raw Whisper large-v3 | Uzbek eval | no FT | 1.0522 | 0.4590 | unusable |
| Mini FT | small subset | partial | 0.4961 | 0.1094 | promising |
| Partial FT USC | USC | encoder 24-31 + decoder | 0.200526 | 0.052908 | protected best |
| Full FT USC | USC | all params | 0.222152 | 0.056583 | worse than partial |

30h proxy highlights:

| Experiment | Strategy | WER | CER |
|---|---|---:|---:|
| `phase4x_encoder_bcd_decoder_2e5_bs4_fast` | A frozen, B/C/D/decoder `2e-5` | 0.191341 | 0.048445 |
| `phase4x_encoder_b1e5_cd_decoder2e5_bs4_fast` | A frozen, B `1e-5`, C/D/decoder `2e-5` | 0.192737 | 0.048256 |
| `phase4x_main_encoder_cd_decoder_5em05_bs4_fast` | encoder 16-31 + decoder `5e-5` | 0.193735 | 0.050009 |
| `phase4x_full_encoder_decoder_1e5_bs1_safe` | all blocks `1e-5` | 0.223464 | 0.056402 |
| `phase4x_full_encoder_decoder_2e5_bs1_safe` | all blocks `2e-5` | 3.911213 | 2.254956 |

## Immediate Next Work

1. Monitor `whisper_gold_ft` through the first validation at step 1000.
2. If validation WER improves, verify `outputs_full_gold/best_model/` and
   `outputs_full_gold/metrics/best_metrics.json`.
3. When the run completes, compare validation trajectory against proxy expectations.
4. Run final locked test evaluation only after the model/checkpoint is selected.
5. Resume Silver teacher scoring after the active Gold run no longer needs the GPU.

## Useful Commands

```bash
cd /home/mahmud/whisper-uz-ft
git status --short
tmux ls
pgrep -af 'src/train.py|run_experiment.py|score_teacher.py|autonomous_search.py'
nvidia-smi
df -h .
python scripts/lr_search/audit_data_leakage.py
python scripts/lr_search/verify_freeze_modes.py
python -m text_normalization.tests
```
