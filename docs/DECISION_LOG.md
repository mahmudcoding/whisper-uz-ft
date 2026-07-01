# Decision Log

Last rebuilt: `2026-07-01T04:50:03Z`.

## Objective

The project optimizes Uzbek-only ASR quality. Catastrophic forgetting of English, Russian, Turkish, and general multilingual Whisper behavior is acceptable if Uzbek WER/CER improves.

## Core Decisions

1. **Use `openai/whisper-large-v3` as the base model.**
   Rationale: strongest open Whisper base available in this project; previous Uzbek failure is attributed to priors/data, not insufficient base capacity.
   Status: active.

2. **Force Uzbek decoding everywhere.**
   Rationale: automatic language detection contributed to Turkish/Kazakh confusion. Training and evaluation use `language=uz`, `task=transcribe`.
   Status: active and enforced in `src/model.py`.

3. **Do not use test data during search or training.**
   Rationale: preserve benchmark integrity. Configs for full Gold and Stage 1 use `load_test_split: false` and `evaluate_test_after_training: false`.
   Status: active.

4. **Protect the partial FT USC baseline.**
   Rationale: it is the best fully evaluated protected baseline on USC test: WER 20.05%, CER 5.29%.
   Artifact: `models/partial_ft_usc_baseline/`.
   Status: never modify/delete.

5. **Reject full encoder FT on small USC-only data as default.**
   Evidence: full FT USC result was WER 22.22%, CER 5.66%, worse than partial FT.
   Status: decision stands.

6. **Use B/C/D encoder blocks plus decoder at 2e-5 for larger Gold and Gold+Silver training.**
   Evidence: best 30h proxy run `phase4x_encoder_bcd_decoder_2e5_bs4_fast` achieved WER 19.13%, CER 4.84%.
   Status: active.

7. **Move FeruzaSpeech from Gold to Silver.**
   Rationale: current Gold must be high-trust only; FeruzaSpeech is useful but lower trust than USC/Common Voice/FLEURS.
   Status: active; val/test have zero Feruza matches.

8. **Use Kotib teacher for Silver scoring, not this project’s fine-tuned model.**
   Rationale: using the same student lineage as teacher creates confirmation bias.
   Evidence: `reports/silver_quality_report/teacher_scoring_config.json` uses `Kotib/uzbek_stt_v1`, revision `0e239511f65c1c7bbf426619a1ee9ea628411344`.
   Status: active.

9. **Disable persistent Hugging Face feature caching for large training.**
   Rationale: previous Stage 1 failed from disk exhaustion at checkpoint save after huge cache/checkpoint growth.
   Status: active in `src/train.py` and Stage 1 no-cache config.
