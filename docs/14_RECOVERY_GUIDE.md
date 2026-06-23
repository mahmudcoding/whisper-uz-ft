# Recovery Guide

This guide assumes everything is lost except `docs/`.

## 1. Recreate Project Directory

```bash
mkdir -p /home/mahmud/whisper-uz-ft
cd /home/mahmud/whisper-uz-ft
```

Restore source code from version control or reconstruct files described in these docs:

- `src/train.py`
- `src/model.py`
- `src/text_normalization/uz_normalizer.py`
- `src/dedup/`
- `src/data_quality/`
- `src/data_sampling/`
- `benchmark/`
- `scripts/`
- `configs/`

## 2. Recreate Python Environment

```bash
cd /home/mahmud/whisper-uz-ft
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers datasets evaluate accelerate soundfile librosa jiwer pyyaml numpy pandas tqdm
python -m pip install faster-whisper ctranslate2 bitsandbytes
```

Set runtime variables:

```bash
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
```

Validate:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY
```

## 3. Rebuild USC Data

Download/stage ISSAI USC under:

```bash
/home/mahmud/datasets/usc/ISSAI_USC
```

Expected final project splits:

- `data/train.csv`: 99,617 rows.
- `data/val.csv`: 3,762 rows.
- `data/test.csv`: 3,821 rows.

Each row should contain:

```text
audio_path,text,duration,speaker_id,split,source_metadata
```

## 4. Rebuild Gold Master Corpus

Stage:

- USC at `/home/mahmud/datasets/usc/ISSAI_USC`.
- Common Voice Uzbek at `/home/mahmud/datasets/common_voice_uz`.
- FLEURS Uzbek at `/home/mahmud/datasets/fleurs_uz`.
- FeruzaSpeech after HF access at `/home/mahmud/datasets/feruzaspeech`.

Use:

```bash
python scripts/download_datasets/prepare_existing_usc.py
python scripts/download_datasets/export_hf_audio_dataset.py --help
```

Recreate:

- `data/gold_work/gold_raw_combined.csv`
- `data/gold_work/gold_hashes.csv`
- `data/gold_work/gold_text_dedup.csv`
- `data/gold_work/gold_overlap.csv`
- `data/gold_work/gold_quality.csv`
- `data/gold_master/train.csv`
- `data/gold_master/val.csv`
- `data/gold_master/test.csv`

Target current Gold size without Feruza:

- 184,140 rows.
- `207.12h`.

## 5. Rebuild Configs

Critical current full FT config:

```yaml
model_name: openai/whisper-large-v3
language: uz
task: transcribe
data_dir: ~/whisper-uz-ft/data
output_dir: ~/whisper-uz-ft/outputs_full_ft
logging_dir: ~/whisper-uz-ft/logs/tensorboard_full_ft
train_last_encoder_blocks: all
require_cuda: true
learning_rate: 8.0e-6
encoder_learning_rate: 2.0e-6
decoder_learning_rate: 8.0e-6
epochs: 1
per_device_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 32
effective_batch: 32
warmup_ratio: 0.1
weight_decay: 0.03
eval_steps: 1000
save_steps: 1000
logging_steps: 25
max_grad_norm: 1.0
scheduler: cosine
bf16: true
fp16: false
gradient_checkpointing: true
dataloader_num_workers: 8
feature_preprocessing_num_workers: 1
early_stopping_patience: 5
metric_for_best_model: wer
greater_is_better: false
save_total_limit: 4
generation_max_length: 225
generation_num_beams: 1
apply_spec_augment: true
```

## 6. Run Sanity Check

```bash
cd /home/mahmud/whisper-uz-ft
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1
python src/train.py --config configs/full_ft_uzbek.yaml --sanity-check --sanity-report logs/full_ft_sanity_report.json
```

## 7. Launch Training

```bash
tmux new-session -d -s whisper_full_ft_uzbek -n train \
  "cd /home/mahmud/whisper-uz-ft; source .venv/bin/activate; export PYTHONPATH=src PYTHONUNBUFFERED=1; python src/train.py --config configs/full_ft_uzbek.yaml --resume auto 2>&1 | tee -a logs/full_ft_uzbek.log"
tmux new-window -t whisper_full_ft_uzbek -n monitor \
  "cd /home/mahmud/whisper-uz-ft; bash scripts/system_monitor.sh logs/full_ft_uzbek_system.log"
```

Monitor:

```bash
tmux attach -t whisper_full_ft_uzbek
tail -f logs/full_ft_uzbek.log
nvidia-smi
```

## 8. Evaluate

After training:

```bash
python benchmark/eval_suite.py --model-path outputs_full_ft/final_model --manifest data/test.csv --output reports/full_ft_eval.json
python benchmark/language_confusion_benchmark.py --model-path outputs_full_ft/final_model --manifest data/test.csv
```

Compare against:

- `partial_ft_usc_baseline`: WER `20.05%`, CER `5.29%`.

## 9. Troubleshooting

CUDA OOM:

- Reduce `per_device_batch_size`.
- Increase gradient accumulation to keep effective batch.
- Ensure gradient checkpointing is enabled.

No checkpoint:

- Inspect `logs/full_ft_uzbek.log`.
- Verify `output_dir`.
- Verify disk space.

Bad WER/hallucinations:

- Confirm forced Uzbek decoding.
- Confirm normalizer in evaluation.
- Inspect sample predictions.

Dataset load failure:

- Verify manifest columns.
- Verify all `audio_path` files exist.
- Run audio validation.
