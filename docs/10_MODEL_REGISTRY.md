# Model Registry

## Registry Rules

Every model or checkpoint used for decisions must be registered here with:

- Registry ID.
- Path.
- Base model.
- Training data.
- Config.
- Trainable parameters.
- Metrics.
- Status.
- Notes.

## Models

### raw_whisper_large_v3

- Base model: `openai/whisper-large-v3`.
- Path: Hugging Face model hub.
- Training: none.
- Forced Uzbek decoding: used in project evaluations where possible.
- Full Uzbek baseline: WER `1.0522`, CER `0.4590`.
- Mini validation baseline: WER `1.3799`, CER `0.8109`.
- Status: baseline only.
- Key issue: severe hallucination and language-prior confusion.

### mini_ft_usc

- Path: `outputs/mini/final_model/`.
- Config: `configs/mini_train.yaml`.
- Dataset: mini USC split.
- Result: WER `0.4960674157`, CER `0.1094431339`.
- Status: experimental smoke model.
- Use: proof that Uzbek adaptation works.

### partial_ft_usc_baseline

- Archive path: `archive/partial_ft_usc/model/`.
- Source path: `outputs/final_model/`.
- Config archive: `archive/partial_ft_usc/config/`.
- Metrics: `archive/partial_ft_usc/metrics/test_metrics.json`.
- Dataset: USC only.
- Training: partial FT.
- Frozen: encoder blocks 0-23.
- Trainable: encoder blocks 24-31 and full decoder.
- Trainable params: `1,063,930,880`.
- Test WER: `0.2005258480`.
- Test CER: `0.0529079419`.
- Status: current best completed model.
- Protection: do not modify archived baseline.

### full_ft_usc_layerwise_current

- Output path: `outputs_full_ft/`.
- Config: `configs/full_ft_uzbek.yaml`.
- Dataset: USC only.
- Training: full FT.
- Trainable params: all `1,543,490,560`.
- Precision: BF16.
- Encoder LR: `2e-6`.
- Decoder LR: `8e-6`.
- Epochs: current intended `1`.
- Status: running/in-progress.
- Status: stopped after resume failure.
- Current known milestone: step 1000 reached with loss `11.7892`.
- Step-1000 validation: WER `0.3332`, CER `0.09192`.
- Final WER/CER: not available yet.
- Important: checkpoint `outputs_full_ft/checkpoint-1000` exists, but resume currently fails until PyTorch is upgraded to `>=2.6` or a documented workaround is chosen.

### gold_master_future

- Output path: not created.
- Dataset: `data/gold_master/`, `207.12h`.
- Training: planned.
- Status: not launched.
- Requirement before launch: adapt training loader/schema and weighted sampler.

## Checkpoint Paths

Partial FT checkpoints:

- `outputs/checkpoint-2500`
- `outputs/checkpoint-3000`
- `outputs/checkpoint-3114`
- Archived copies under `archive/partial_ft_usc/checkpoints/`.

Dry-run checkpoints:

- `outputs_full_ft_dry_run/checkpoint-50`
- `outputs_full_ft_dry_run/checkpoint-100`

Full FT current:

- `outputs_full_ft/` contains run metadata.
- `outputs_full_ft/checkpoint-1000` exists and includes optimizer/scheduler state.
