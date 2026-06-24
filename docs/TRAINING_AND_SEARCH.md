# Training and Learning-Rate Search

**Document role:** Authoritative training behavior, configuration contracts, tuning
modes, safety controls, LR-search protocol, and promotion criteria.

## Training Entry Points

| Purpose | Entry point |
|---|---|
| Core Trainer | `src/train.py` |
| Model loading/freezing | `src/model.py` |
| Partial FT config | `configs/train.yaml` |
| Completed full-FT config | `configs/full_ft_uzbek.yaml` |
| LR-search configs | `configs/lr_search/` |
| One LR experiment | `scripts/lr_search/run_experiment.py` |
| Autonomous LR search | `scripts/lr_search/autonomous_search.py` |
| Freeze verification | `scripts/lr_search/verify_freeze_modes.py` |

## Uzbek-Only Model Contract

Every training/evaluation model is loaded with:

```yaml
language: uz
task: transcribe
```

The processor supplies forced decoder prompt IDs. Automatic language detection is not
used for Uzbek evaluation.

## Supported Tuning Modes

Whisper large-v3 has 32 encoder blocks.

| Mode | Encoder state | Decoder state | Trainable parameters |
|---|---|---|---:|
| `decoder_only` | all frozen | trainable | 906,521,600 |
| `encoder_24_31_plus_decoder` | 0-23 frozen | trainable | 1,063,930,880 |
| `encoder_16_31_plus_decoder` | 0-15 frozen | trainable | 1,221,340,160 |
| full FT | all trainable | trainable | 1,543,490,560 |

Verify actual parameter state:

```bash
source .venv/bin/activate
export PYTHONPATH=src
python scripts/lr_search/verify_freeze_modes.py
```

## Optimizer and Precision

`src/train.py` builds AdamW groups with:

- encoder LR from `encoder_learning_rate`;
- decoder/output LR from `decoder_learning_rate`;
- configured weight decay for standard weights;
- zero decay for bias and layer-norm weights;
- tied-parameter deduplication.

For large-v3 on the A40:

- use BF16;
- use per-device batch 1 unless measured otherwise;
- preserve effective batch through gradient accumulation;
- enable gradient checkpointing;
- use `max_grad_norm: 1.0`.

FP16 remains part of the historical partial-FT baseline, not the default for new search
runs.

## Dataset Loading

`src/train.py` expects:

```text
audio_path,text,duration,speaker_id,split,source_metadata
```

The loader:

1. reads train and validation manifests;
2. optionally loads test;
3. decodes/resamples audio to 16 kHz;
4. extracts Whisper features;
5. tokenizes labels;
6. removes source columns after preprocessing.

During LR search, `load_test_split: false` prevents test data from being read or
feature-preprocessed.

## Metrics

Training evaluation emits:

- `eval_loss`;
- `eval_wer`;
- `eval_cer`;
- `eval_hallucination_rate`;
- `eval_language_confusion_rate`.

Test evaluation, when explicitly enabled after locking a configuration, emits
`test_*` metrics. Never use `test_wer` for early stopping or model selection.

WER/CER are ratios: `0.20` means 20%.

## Safety Controls

`SafetyCallback`:

- stops on non-finite loss;
- tracks repeated unsafe gradient norms;
- requests save before stopping.

`ProductionStatusCallback`:

- writes milestone and evaluation reports;
- captures GPU memory/utilization;
- estimates ETA;
- records sample predictions;
- tracks WER regression;
- tracks hallucination growth.

Checkpoint validation checks nonempty model weights, `trainer_state.json`, and
`training_args.bin`. Trainer checkpoints also preserve optimizer and scheduler state.

## Resume Contract

```bash
python src/train.py --config <resolved-config> --resume auto
```

`auto` selects the highest `checkpoint-N` in the configured output directory.

Resume is valid only when the checkpoint matches:

- model architecture and tuning mode;
- optimizer groups;
- precision;
- dataset;
- scheduler and total-step assumptions.

Do not silently resume with a different experiment ID or resolved config.

## LR-Search Design

### Proxy Data

- coarse: approximately 10 training hours;
- main: approximately 30 training hours;
- source mix: 50% USC, 40% Common Voice, 10% FLEURS;
- deterministic seeds;
- validation-only selection;
- locked, unloaded test manifests.

### Practical Tie Thresholds

- WER delta below `0.003`;
- CER delta below `0.001`.

Values inside either threshold require statistical judgment. Prefer lower LR when
quality and stability are effectively tied.

### Phase 1A: Divergence Screen

Mode: decoder-only.  
Dataset: coarse proxy.  
Steps: 300.

Decoder LRs:

- `2e-6`
- `8e-6`
- `2e-5`
- `5e-5`

Reject on NaN/Inf, loss divergence, severe hallucination/confusion, missing validation,
checkpoint failure, or runner instability.

### Phase 1B: Full Coarse Runs

Run only Phase 1A survivors for the complete coarse schedule. Rank by:

1. validation WER;
2. validation CER;
3. stability and convergence.

Promote the top two rather than only the numerical winner.

### Phase 2A: Main Decoder Confirmation

Run both promoted decoder LRs on the 30h proxy. Do not overfit to small deltas.

### Phase 2B: Upper Encoder LR

Fix the selected decoder LR and train encoder 24-31 with:

- `5e-7`
- `1e-6`
- `2e-6`
- `5e-6`

Include decoder-only as the control.

### Phase 3: Freeze Boundary

Compare on the same proxy:

- decoder-only;
- encoder 24-31 + decoder;
- encoder 16-31 + decoder.

Full FT is the historical control and is currently disfavored by measured USC results.

## Running the Search

One dry run:

```bash
python scripts/lr_search/run_experiment.py \
  --config configs/lr_search/phase1a_decoder_lr_2e6.yaml \
  --dry-run
```

Persistent controller:

```bash
tmux new-session -d -s whisper_lr_search \
  "cd /home/mahmud/whisper-uz-ft && source .venv/bin/activate && \
   export PYTHONPATH=src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && \
   python scripts/lr_search/autonomous_search.py \
   2>&1 | tee -a reports/lr_search/autonomous_search_console.log"
```

Never start this command if `tmux has-session -t whisper_lr_search` succeeds.

## Experiment Output Contract

Each `outputs_lr_search/<experiment_id>/` contains:

- resolved `config.yaml`;
- `experiment.json`;
- `train.log`;
- `gpu_metrics.csv`;
- `trainable_parameters.json`;
- `trainable_parameter_groups.json`;
- checkpoints;
- `run_metrics.json`;
- aggregated `metrics.json`;
- plots.

Search aggregation refuses outputs containing test metrics.

## Promotion to 207h Gold

After selecting a stable validation winner:

1. lock all hyperparameters;
2. verify test hashes;
3. evaluate the proxy test once;
4. convert Gold master to the training schema;
5. run a Gold sanity check;
6. start with one Gold epoch;
7. select checkpoints using Gold validation only;
8. compare final test quality against `partial_ft_usc_baseline`.

Do not launch the 207h job until the search has produced
`reports/lr_search/FINAL_RECOMMENDATION.md`.
