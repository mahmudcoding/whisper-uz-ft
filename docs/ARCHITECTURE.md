# System Architecture

**Document role:** Stable technical reference for components, contracts, and artifact
flow. Runtime status belongs in `STATUS.md`.

## End-to-End System

```text
External datasets
    |
    v
Acquisition and export
    |
    v
16 kHz mono audio + source manifests
    |
    v
Canonical Uzbek normalization
    |
    v
Validation -> deduplication -> quality scoring
    |
    v
Leakage-controlled train / validation / test manifests
    |
    +--> Proxy subset builder --> LR/freeze search
    |
    v
Whisper large-v3 fine-tuning
    |
    v
Validation selection -> locked test evaluation
    |
    +--> Model registry
    +--> Error analysis
    +--> faster-whisper conversion
    +--> Inference benchmark and capacity planning
```

## Component Boundaries

### Data Acquisition

Owned by `scripts/download_datasets/`.

Responsibilities:

- download or export accessible datasets;
- decode audio without depending on optional TorchCodec paths;
- convert to mono 16 kHz;
- preserve source, split, speaker, transcript, and licensing context;
- write deterministic manifests.

Raw/staged audio lives under `/home/mahmud/datasets/`, not in Git.

### Text Normalization

Owned by `src/text_normalization/`.

Responsibilities:

- Unicode NFKC normalization;
- apostrophe canonicalization;
- Uzbek Cyrillic-to-Latin transliteration;
- punctuation and whitespace cleanup;
- optional punctuation removal.

The canonical interface is:

```python
from text_normalization import normalize_uzbek_text
```

### Deduplication

Owned by `src/dedup/`.

Signals:

- exact and near audio hash;
- transcript equality and near similarity;
- duration agreement;
- cross-dataset overlap.

Audio duplicates can be removed after review. Transcript duplicates are evidence, not
automatic removal criteria, because read-speech corpora legitimately repeat phrases.

### Quality Scoring and Filtering

Owned by `src/data_quality/` and `src/filtering/`.

Signals include duration, transcript length, character rate, symbols, audio statistics,
and optional teacher WER/CER. Outputs classify rows as keep, suspicious, or reject.

### Training

Owned by `src/model.py` and `src/train.py`.

`src/model.py`:

- loads processor/model;
- forces Uzbek transcription;
- applies freeze boundaries;
- enables gradient checkpointing;
- reports trainable parameter groups.

`src/train.py`:

- loads manifests and audio;
- constructs training/evaluation datasets;
- builds layer-wise AdamW groups;
- computes WER/CER and safety indicators;
- manages checkpointing, resume, early stopping, telemetry, and final evaluation.

### LR Search

Owned by `scripts/lr_search/` and `configs/lr_search/`.

- `create_lr_subsets.py`: deterministic duration/source-stratified subsets.
- `validate_lr_subsets.py`: hours, composition, paths, and speaker checks.
- `audit_data_leakage.py`: split and test-isolation audit.
- `run_experiment.py`: one reproducible run with resolved config and GPU telemetry.
- `autonomous_search.py`: phase orchestration, resume, ranking, and report generation.
- `compare_experiments.py`: validation-only ranking and plots.

### Evaluation

Owned by `benchmark/eval_suite.py`,
`benchmark/language_confusion_benchmark.py`, and Trainer evaluation in `src/train.py`.

Model selection uses validation. The test set is consumed only after a configuration is
locked.

### Inference and Capacity Planning

Owned by `benchmark/`.

Supported paths:

- Hugging Face Transformers;
- faster-whisper;
- CTranslate2 conversion.

The benchmark runner samples GPU, CPU, and RAM, then writes measured RTF, throughput,
latency, quality, and cost/capacity reports.

## Persistent Artifact Contracts

| Artifact | Location |
|---|---|
| Protected best model | `archive/partial_ft_usc/model/` |
| Gold manifests | `data/gold_master/` |
| LR proxy manifests | `data/lr_search/` |
| Full-FT output | `outputs_full_ft/` |
| LR-search outputs | `outputs_lr_search/<experiment_id>/` |
| Training logs | `logs/` or run-local `train.log` |
| Search reports | `reports/lr_search/` |
| Quality/dedup reports | `reports/gold_*_report/` |
| Inference benchmark results | `benchmark/results/` |
| Capacity reports | `benchmark/reports/` |

Each LR experiment directory contains its resolved `config.yaml`, immutable experiment
metadata, log, GPU metrics, checkpoints, final model, and structured metrics.

## Runtime Topology

Long jobs run in tmux. The active LR search uses one controller process, one experiment
runner, one Trainer parent, and DataLoader workers. Multiple `src/train.py` PIDs can
therefore be expected; verify parent/child relationships before treating them as
duplicate training jobs.

Only one process may own a given output directory.
