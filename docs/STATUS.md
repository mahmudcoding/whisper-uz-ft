# Current Status

Last verified: 2026-06-28T16:56:49Z

This file answers what is true now. If it conflicts with logs, manifests, process
state, or metrics files, inspect the live artifacts and update this file.

## Executive State

- Mission: build the best open-weight Uzbek ASR model with Whisper large-v3.
- Optimization target: Uzbek WER/CER only; multilingual retention is not required.
- Base model: `openai/whisper-large-v3`.
- Best completed locked-test model: `models/partial_ft_usc_baseline/model/`.
- Best completed locked-test metrics: WER `0.2005258480`, CER `0.0529079419`.
- Best LR-search proxy configuration: `phase4x_encoder_bcd_decoder_2e5_bs4_fast`.
- Active training: full Gold one-epoch run using the best proxy configuration.
- Active tmux session: `whisper_gold_ft`.
- No LR-search controller is currently running.
- Test data is not loaded during the active full Gold run.

## Active Runtime

Full Gold fine-tuning is running in tmux.

| Field | Value |
|---|---|
| tmux session | `whisper_gold_ft` |
| command | `.venv/bin/python src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml --resume auto` |
| started | 2026-06-28 16:30 UTC |
| output root | `outputs_full_gold/` |
| log | `outputs_full_gold/logs/full_gold_training.log` |
| step metrics | `outputs_full_gold/logs/training_metrics.jsonl` |
| status reports | `outputs_full_gold/status_reports/` |
| checkpoints | `outputs_full_gold/checkpoints/` |
| best model snapshot | `outputs_full_gold/best_model/` |
| best metrics | `outputs_full_gold/metrics/best_metrics.json` |
| config snapshot | `outputs_full_gold/config_snapshot.yaml` |

Latest inspected state at 2026-06-28T16:56:49Z:

- progress: about step `345 / 5380`;
- latest JSON metric: step `340`, loss `3.2627`, grad norm `30.32`, LR `1.260e-5`;
- first validation has not run yet;
- no checkpoint has been saved yet;
- GPU: NVIDIA A40, about `38.0 GiB / 46.1 GiB` used, about `100%` utilization;
- first validation and first scheduled checkpoint are expected at step `1000`.

Monitor:

```bash
cd /home/mahmud/whisper-uz-ft
tmux attach -t whisper_gold_ft
tail -f outputs_full_gold/logs/full_gold_training.log
tail -f outputs_full_gold/logs/training_metrics.jsonl
nvidia-smi
```

Do not start another large training job while this process is active.

## Active Full Gold Config

Config: `configs/full_training/gold_bcd_decoder_2e5.yaml`.

| Setting | Value |
|---|---|
| Dataset | `data/gold_master_training_schema/` |
| Train rows/hours | 172,135 / 186.4037h |
| Validation rows/hours | 6,068 / 10.3556h |
| Test loading | disabled |
| Epochs | 1 |
| Per-device batch | 4 |
| Gradient accumulation | 8 |
| Effective batch | 32 |
| Optimizer steps | 5,380 |
| Precision | BF16 |
| Gradient checkpointing | disabled |
| Scheduler | cosine |
| Warmup ratio | 0.10 |
| Eval/save steps | 1000 / 1000 |
| Early stopping | validation WER patience 3 |
| Metric for best model | validation WER, lower is better |

Trainable groups:

| Group | State | LR | Parameters |
|---|---|---:|---:|
| Encoder 0-7 | frozen | 0 | 0 / 157,409,280 |
| Encoder 8-15 | trainable | 2e-5 | 157,409,280 |
| Encoder 16-23 | trainable | 2e-5 | 157,409,280 |
| Encoder 24-31 | trainable | 2e-5 | 157,409,280 |
| Decoder + proj_out | trainable | 2e-5 | 906,521,600 |

Total trainable parameters: `1,378,749,440` of `1,543,490,560` (`89.3267%`).

## Current Best Evidence

### Protected Baseline

The best completed model with locked test metrics remains the USC partial fine-tune:

- path: `models/partial_ft_usc_baseline/model/`;
- dataset: USC only;
- frozen: encoder 0-23;
- trainable: encoder 24-31 + decoder;
- precision: FP16;
- epochs: 1;
- test WER: `0.2005258480`;
- test CER: `0.0529079419`.

This artifact is protected. Do not modify or delete it.

### Best LR-Search Proxy

Best 30h validation proxy result:

- experiment: `phase4x_encoder_bcd_decoder_2e5_bs4_fast`;
- data: `data/lr_search/main_30h/`;
- epochs: 2;
- batch/accum: 4 / 8;
- trainable: encoder 8-31 + decoder;
- block A LR: 0;
- block B/C/D LR: `2e-5`;
- decoder LR: `2e-5`;
- best validation WER: `0.1913407821`;
- best validation CER: `0.0484449599`;
- best step: `1642`;
- peak observed VRAM: about `38.1 GiB`.

This is proxy evidence, not a locked final test result. The active full Gold run is the
promotion test of this configuration.

## Dataset State

Gold master:

- path: `data/gold_master/`;
- training schema: `data/gold_master_training_schema/`;
- sources: USC, Common Voice Uzbek, FLEURS Uzbek;
- total: 184,140 rows / 207.1150h;
- train: 172,135 rows / 186.4037h;
- validation: 6,068 rows / 10.3556h;
- test: 5,937 rows / 10.3557h;
- validation report: no missing paths, no path/hash leakage, no known speaker leakage.

Silver:

- finalized Silver master: `data/silver_master/`;
- finalized Silver source: FeruzaSpeech only, train-only, 12,854 rows / 57.8279h;
- large Silver sources are exported and prefiltered but not finalized:
  UzbekVoice filtered, IT YouTube, News YouTube, Tashkent podcasts;
- Kotib teacher scoring was stopped at 129,007 scored candidates out of 535,728
  prefiltered candidates.

FeruzaSpeech is not Gold. It was moved out of Gold on 2026-06-27 because of gated,
restrictive licensing.

## Storage and Worktree

Project directory size at verification: about `33G`.
Filesystem `/dev/vda1`: 2.0T total, 856G used, 1.2T available, 44% used.

The worktree is intentionally dirty from generated artifacts, docs, LR-search configs,
and cleaned old LR-search checkpoints. Do not use `git reset --hard`.

## Immediate Next Actions

1. Let `whisper_gold_ft` continue unless NaN, OOM, repeated WER regression, or severe
   hallucination occurs.
2. At step 1000, inspect `outputs_full_gold/status_reports/status_step_1000_eval.*`
   and `outputs_full_gold/metrics/best_metrics.json`.
3. When training completes, evaluate the selected best model according to the locked
   final evaluation protocol.
4. Only after the Gold run is understood, resume Silver teacher scoring with
   `Kotib/uzbek_stt_v1`; do not use an in-project fine-tuned model as Silver teacher.
