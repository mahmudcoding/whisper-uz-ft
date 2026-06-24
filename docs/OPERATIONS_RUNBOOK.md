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
pgrep -af 'train.py|run_experiment.py|autonomous_search.py'
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

## Active LR Search

Session:

```text
whisper_lr_search
```

Noninteractive monitoring:

```bash
tmux has-session -t whisper_lr_search
tail -f reports/lr_search/autonomous_search_console.log
tail -f reports/lr_search/autonomous_search.log
nvidia-smi
```

Inspect active experiment:

```bash
pgrep -af 'run_experiment.py|src/train.py'
find outputs_lr_search -maxdepth 2 -name metrics.json -printf '%T@ %p\n' | sort -n
```

Multiple `src/train.py` PIDs can be DataLoader workers. Use process ancestry before
concluding that duplicate training is running:

```bash
ps -eo pid,ppid,cmd --forest | rg 'autonomous_search|run_experiment|src/train'
```

## Persistent Launch

Only if the session does not exist:

```bash
tmux new-session -d -s whisper_lr_search \
  "cd /home/mahmud/whisper-uz-ft && source .venv/bin/activate && \
   export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   python scripts/lr_search/autonomous_search.py \
   2>&1 | tee -a reports/lr_search/autonomous_search_console.log"
```

Do not run a second controller against the same `outputs_lr_search/`.

## Resume

### Autonomous Controller

The controller reuses completed metrics and resumes incomplete experiments when a valid
checkpoint exists. Relaunch the controller only after confirming no old controller or
Trainer process remains.

### Single Experiment

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
find outputs_lr_search/<id>/checkpoint-<step> -maxdepth 1 -type f -printf '%f %s\n'
python -m json.tool outputs_lr_search/<id>/checkpoint-<step>/trainer_state.json >/dev/null
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
