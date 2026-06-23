# Experiment Log

## 2026-06-23

- Audited the active full training session `whisper_full_training`.
- Confirmed the one-epoch partial fine-tuning run completed successfully.
- Recorded final test metrics: WER 0.2005258480, CER 0.0529079419.
- Confirmed final model exists at `outputs/final_model` and final checkpoint exists at `outputs/checkpoint-3114`.
- Created timestamped backups before modifying filtering code.
- Implemented production Uzbek text normalization in `src/text_normalization/`.
- Implemented reusable quality scoring/filtering in `src/filtering/`.
- Implemented reproducible evaluation CLI in `benchmark/eval_suite.py`.
- Fixed dataset quality scoring to normalize text before suspicious-symbol detection.
- Regenerated dataset audit outputs: `reports/dataset_quality_scores.csv` and `bad_samples.csv`.
- Ran normalizer tests successfully: 5 tests passed, about 2.06M chars/sec.
- Verified Python compile checks for new normalization, filtering, and evaluation modules.
- Verified A40 BF16 support with PyTorch and actual Whisper forward passes.
- Created project documentation set under `docs/`.


- Patched `benchmark/eval_suite.py` to support manifest alias, batch size, precision selection, throughput, and peak VRAM reporting.
- Fixed `benchmark/eval_suite.py` to load audio through `soundfile`/`librosa` instead of requiring system `ffmpeg`.
- Ran one-sample evaluation smoke test successfully and wrote `reports/eval_suite_smoke.json`.

- Re-audited project for Uzbek-only objective; added full fine-tuning config, forced-Uzbek decoding patches, and language-confusion benchmark.
- Ran full-FT sanity check with `configs/full_ft_uzbek.yaml`: all 1.543B parameters trainable, CUDA forward pass ok.
- Ran 20-sample teacher subset evaluation with final model: WER 0.0782, CER 0.0139.
- Ran corrected language-confusion smoke benchmark: zero Turkish/Kazakh/Russian/English leakage on 3 samples.

- Archived the completed partial FT baseline to immutable reference path `archive/partial_ft_usc/`; copied final model, metrics, configs, logs, and retained checkpoints without deleting originals.
- Generated `archive/partial_ft_usc/BASELINE_REPORT.md`; archived baseline metrics are WER `0.2005258480`, CER `0.0529079419`.
- Deep-mined `/home/mahmud/rubaistt_v2-open-sourced` and wrote `docs/RUBAI_DEEP_MINING.md`; highest-impact findings are data diversity, pseudo-labeling with context, ASR/similarity filtering, strong Uzbek normalization, full FT BF16, SpecAugment, and robust segmentation.
- Reviewed and updated full-FT planning docs: `docs/FULL_FT_CONFIG_REVIEW.md`, `docs/HYPERPARAMETER_RECOMMENDATION.md`, `docs/FINETUNING_STRATEGY_COMPARISON.md`, and `docs/MAX_PERFORMANCE_PLAN.md`.
- Ran BF16 full fine-tuning dry run with `configs/full_ft_dry_run.yaml` for 100 steps on mini split. The run completed successfully, saved checkpoints at steps 50 and 100, and wrote final test metrics.
- Dry-run evidence: loss dropped from `28.31` to `5.269`, peak VRAM was about `30017.736 MiB`, final mini test WER was `0.5567415730`, and final mini test CER was `0.1356780327`.
- Decision gate recorded in `docs/DRY_RUN_REPORT.md`: launch current full-FT config only after user approval; no long training launched.

- User rejected immediate uniform full-FT launch and requested one more optimization pass.
- Reduced full-FT evaluation/checkpoint cadence from 500 to 1000 steps in `configs/full_ft_uzbek.yaml`, cutting estimated 4-epoch runtime from about 74-80h to about 65-68h.
- Implemented layer-wise LR support in `src/train.py`: encoder LR `2e-6`, decoder LR `8e-6`, PyTorch AdamW param groups, zero weight decay for bias/layer norm, and optimizer group JSON reporting.
- Made best-model metric explicit in config: `metric_for_best_model: wer`, `greater_is_better: false`.
- Removed `EarlyStoppingCallback` before final test evaluation so test metrics remain `test_wer`/`test_cer` without early-stopping metric mismatch.
- Validated layer-wise LR sanity initialization on CUDA with `configs/full_ft_dry_run.yaml`; optimizer groups covered all 1,543,490,560 trainable parameters and one forward pass completed successfully.

- Approval granted for Experiment B: layer-wise LR full fine-tuning.
- Patched `configs/full_ft_uzbek.yaml` to enforce `max_grad_norm: 1.0` and configured status reports at steps 100, 500, 1000, and every evaluation.
- Added production status reporting and stop-condition callbacks to `src/train.py`: milestone/eval reports, sample predictions, language-confusion indicators, hallucination indicators, WER two-eval regression stop, and hallucination-rate jump stop.
- Ran full-config sanity check before launch: 99,617 train rows, 3,762 validation rows, 3,821 test rows, all 1,543,490,560 params trainable, CUDA forward pass ok.
- Launched training in tmux session `whisper_full_ft_uzbek` at 2026-06-23T06:18:33Z.
- Training log: `logs/full_ft_uzbek.log`; system monitor log: `logs/full_ft_uzbek_system.log`; status reports: `logs/full_ft_status_reports/`.

- Audited local dataset inventory for max-scale data pipeline. Only USC is currently staged locally under `/home/mahmud/datasets/usc`; Common Voice, FeruzaSpeech, FLEURS, UzbekVoice, YouTube/news/podcast, and bronze corpora are missing locally.
- Added dry-run-first dataset acquisition/prep scripts under `scripts/download_datasets/`.
- Added dedup pipeline modules under `src/dedup/`: audio hashing, transcript dedup, and dataset-overlap detection.
- Added sample-level quality scoring under `src/data_quality/` with text, duration, silence, SNR proxy, and optional teacher metrics.
- Added trust-weighted sampling utilities under `src/data_sampling/`.
- Created data strategy docs: `docs/DATASET_ACQUISITION_PLAN.md`, `docs/DEDUP_STRATEGY.md`, `docs/DATA_NORMALIZATION_PIPELINE.md`, `docs/CURRICULUM_TRAINING_PLAN.md`, `docs/MISSING_DATA_STRATEGY.md`, and `docs/MASTER_DATA_PIPELINE_PLAN.md`.
- Validated data pipeline on USC mini test split. Smoke outputs are in `reports/data_pipeline_smoke/`; canonical USC gold path produced 279 rows, all tagged tier `gold`.

- Executed Gold dataset acquisition and staging. USC was reused from `/home/mahmud/datasets/usc/ISSAI_USC`; FLEURS Uzbek was downloaded from `google/fleurs` config `uz_uz`; Common Voice Uzbek was exported from the accessible cleaned mirror `yakhyo/mozilla-common-voice-uzbek`.
- FeruzaSpeech acquisition is blocked because `k2speech/FeruzaSpeech` is gated manual on Hugging Face and no `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is configured.
- Exported Common Voice Uzbek and FLEURS Uzbek to 16 kHz mono WAV with canonical manifests under `/home/mahmud/datasets/common_voice_uz` and `/home/mahmud/datasets/fleurs_uz`.
- Built combined Gold working manifest `data/gold_work/gold_raw_combined.csv` with 184,325 rows and 207.27h before filtering.
- Ran Gold dedup and quality scoring across USC, Common Voice Uzbek, and FLEURS Uzbek. Removed 50 quality rejects and 135 exact/near audio duplicate rows.
- Built final Gold master corpus at `data/gold_master/`: 184,140 rows, 207.12h total; train 186.40h, validation 10.36h, test 10.36h.
- Validated Gold master split integrity: 0 missing audio paths, 0 duplicate content hashes across splits, and 0 known speaker leakage across train/validation/test.
- Wrote final execution report `docs/GOLD_CORPUS_REPORT.md`; detailed summaries are in `reports/gold_quality_report/` and `reports/gold_dedup_report/`.
