# Operations Runbook

**Document role:** Launch, monitor, resume, stop, and troubleshoot long-running jobs.

## Preflight

Before any training or benchmark:

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

git status --short
tmux ls
pgrep -af 'train.py|run_experiment.py|autonomous_search.py|score_teacher.py'
nvidia-smi
df -h .
.venv/bin/pip check
```

Confirm:

- no conflicting GPU job;
- intended output directory is not owned by another process;
- disk can hold checkpoints;
- config points to the intended manifests;
- epoch count, precision, LRs, freeze mode, eval/save cadence, and test policy are
  correct.

## Active Full Gold Training

Session:

```text
whisper_gold_ft
```

Noninteractive monitoring:

```bash
tmux has-session -t whisper_gold_ft
tail -f outputs_full_gold/logs/full_gold_training.log
tail -f outputs_full_gold/logs/training_metrics.jsonl
nvidia-smi
```

Inspect active training:

```bash
pgrep -af 'src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml'
find outputs_full_gold/checkpoints -maxdepth 1 -type d -name 'checkpoint-*' | sort
ls -lah outputs_full_gold/metrics outputs_full_gold/best_model 2>/dev/null || true
```

Multiple `src/train.py` PIDs can be DataLoader workers. Use process ancestry before
concluding that duplicate training is running:

```bash
ps -eo pid,ppid,cmd --forest | rg 'autonomous_search|run_experiment|score_teacher|src/train'
```

## Persistent Launch: Full Gold

Only if the session does not exist:

```bash
tmux new-session -d -s whisper_gold_ft \
  "cd /home/mahmud/whisper-uz-ft && \
   export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   .venv/bin/python src/train.py --config configs/full_training/gold_bcd_decoder_2e5.yaml --resume auto \
   2>&1 | tee -a outputs_full_gold/logs/full_gold_training.log"
```

Do not run a second trainer against the same `outputs_full_gold/`.

## Resume

### Active Full Gold Run

Resume with the same config and output directory:

```bash
.venv/bin/python src/train.py \
  --config configs/full_training/gold_bcd_decoder_2e5.yaml \
  --resume auto
```

Resume only if tuning mode, optimizer groups, precision, dataset, scheduler, and output
directory are unchanged.

### Historical LR Search

LR search is not active as of 2026-06-28. If intentionally resumed later, use a unique
session and confirm `whisper_gold_ft` is not consuming the GPU.

The autonomous controller reuses completed metrics and resumes incomplete experiments
when a valid checkpoint exists. Relaunch it only after confirming no old controller or
Trainer process remains.

Single LR experiment resume:

```bash
python scripts/lr_search/run_experiment.py \
  --config configs/lr_search/<config>.yaml \
  --experiment-id <existing-id> \
  --resume auto
```

Resume requirements:

- same resolved config;
- same tuning mode and model;
- same dataset;
- compatible optimizer groups;
- valid model, optimizer, scheduler, and trainer state.

## Checkpoint Inspection

Expected Trainer checkpoint contents:

- model weights;
- `trainer_state.json`;
- `training_args.bin`;
- `optimizer.pt`;
- `scheduler.pt`;
- RNG state.

Inspect:

```bash
find outputs_full_gold/checkpoints/checkpoint-<step> -maxdepth 1 -type f -printf '%f %s\n'
python -m json.tool outputs_full_gold/checkpoints/checkpoint-<step>/trainer_state.json >/dev/null
```

The training code refuses known-corrupt checkpoints.

## Safe Stop

Prefer allowing the current optimizer step or evaluation to finish. If an immediate stop
is required:

1. identify the tmux session and parent controller;
2. record current step and latest checkpoint;
3. send `SIGINT` to the owning process or stop the tmux pane;
4. wait for child processes to exit;
5. verify GPU memory is released;
6. preserve logs/output; do not delete the run directory.

Never use `kill -9` unless graceful termination has failed and the process is stuck.

## Health Signals

Healthy training:

- log step advances;
- loss is finite;
- GPU utilization/power is active during training;
- evaluation progresses through validation rows;
- checkpoint files appear at configured steps;
- no safety-stop message.

Healthy evaluation can be slower than training because generation is autoregressive.
Low GPU utilization during CPU feature preprocessing is expected.

## Stop and Investigate

- NaN/Inf loss.
- CUDA OOM.
- repeated unsafe gradient norms.
- WER worsens for two consecutive evaluations.
- hallucination rate rises materially.
- test manifest hash changes.
- LR-search output contains test metrics.
- log and checkpoint timestamps stop while GPU is idle.
- disk approaches checkpoint requirements.
- checkpoint verification fails.

## Common Incidents

### CUDA OOM

1. Check for another GPU process.
2. Capture allocated/reserved/peak VRAM from the error.
3. Keep per-device batch 1.
4. Increase gradient accumulation only to preserve effective batch.
5. Ensure BF16 and gradient checkpointing are enabled.
6. Resume from the latest verified checkpoint.

### Resume Security Error

PyTorch 2.5.1 cannot load optimizer/scheduler state under the installed Transformers
safety requirement. Restore the pinned environment with PyTorch 2.7.1+cu126. Do not
bypass the safety check.

### Dataset Load Failure

- verify manifest schema;
- verify absolute audio paths;
- run subset or Gold validation;
- inspect the first rows and decoded durations;
- confirm the expected split was selected.

### Evaluation Appears Stuck

Check the generation progress bar, GPU power, process state, and log timestamp.
Validation of approximately 845 rows can take around 12-13 minutes on the current
large-v3 search setup.

### Disk Pressure

```bash
df -h /
du -sh outputs* archive benchmark/results
find outputs_lr_search -maxdepth 2 -type d -name 'checkpoint-*' -printf '%p\n'
```

Do not delete the protected baseline. Archive or remove only artifacts whose experiment
and retention status are documented.

## Post-Run Checklist

1. Verify `metrics.json`.
2. Verify no test metrics were produced during search.
3. Verify test hashes.
4. Run experiment comparison.
5. Update `STATUS.md`, `state.json`, the experiment ledger, and model registry.
6. Record failures or decisions.
7. Run `python scripts/update_docs.py --check`.
