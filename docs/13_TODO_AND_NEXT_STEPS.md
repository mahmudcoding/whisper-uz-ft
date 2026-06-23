# TODO and Next Steps

## P0: Immediate

1. Resolve full FT resume failure.
   - Current blocker: Transformers `5.12.1` requires PyTorch `>=2.6` to load checkpoint `.pt` optimizer/scheduler state.
   - Local PyTorch: `2.5.1+cu121`.
   - Preferred action: upgrade PyTorch in `.venv`, then resume from `outputs_full_ft/checkpoint-1000`.

2. Finish one epoch of full FT on USC.
   - Monitor `logs/full_ft_uzbek.log`.
   - Monitor `logs/full_ft_uzbek_system.log`.
   - Check status reports in `logs/full_ft_status_reports/`.

3. Evaluate the full FT checkpoint.
   - Compare against `partial_ft_usc_baseline`.
   - Required metrics: WER, CER, normalized WER/CER, hallucination rate, language-confusion rate.

4. Update docs immediately after evaluation.
   - `01_CURRENT_STATE.md`
   - `08_EXPERIMENT_HISTORY.md`
   - `10_MODEL_REGISTRY.md`
   - `11_DECISIONS_AND_RATIONALE.md` if strategy changes.

## P1: Next Experiment

1. Prepare Gold master training.
   - Adapt `src/train.py` or manifests for `data/gold_master/` schema.
   - Add weighted sampling so Gold subsets are balanced.

2. Acquire FeruzaSpeech.
   - Accept HF gated terms for `k2speech/FeruzaSpeech`.
   - Export to `/home/mahmud/datasets/feruzaspeech`.
   - Rebuild Gold master corpus.

3. Train on Gold master.
   - Start with one epoch.
   - Use layer-wise LR and BF16.
   - Evaluate against USC-only full FT and partial FT baseline.

## P2: Scale Data

1. Acquire Silver datasets:
   - UzbekVoice filtered.
   - IT YouTube Uzbek Speech.
   - News YouTube Uzbek Speech.
   - Podcasts Tashkent Dialect.

2. Run strict filtering:
   - Dedup against Gold.
   - Teacher ASR similarity.
   - WER/CER agreement.
   - Audio quality and SNR proxy.

3. Build curriculum:
   - Stage 1: Gold only.
   - Stage 2: Gold + Silver.
   - Stage 3: Gold + Silver + Bronze.
   - Stage 4: production-domain adaptation.

## P3: Production Readiness

1. Run long-form inference benchmarks on the best checkpoint.
2. Convert best model to faster-whisper/CTranslate2 if quality is preserved.
3. Build streaming benchmark separate from offline throughput.
4. Update capacity planner using measured best-checkpoint throughput.

## Do Not Do Yet

- Do not train on Silver/Bronze before current full FT is evaluated.
- Do not overwrite archived partial FT baseline.
- Do not assume Gold master can be used directly in `src/train.py` without schema check.
- Do not trust raw hours without dedup and quality scoring.
- Do not restart full FT blindly until checkpoint resume compatibility is fixed.
