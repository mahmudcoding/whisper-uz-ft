# Master Uzbek ASR Data Pipeline Plan

Generated: 2026-06-23 UTC

## Executive Goal

Build a maximum-scale Uzbek ASR corpus for Whisper large-v3 that minimizes Uzbek WER/CER. Multilingual preservation is not a goal.

## Current Status

- Current running experiment: full FT Whisper large-v3, layer-wise LR, USC only.
- Best completed baseline: partial FT USC, WER `20.05%`, CER `5.29%`.
- Local data available now: USC only, ~104.63h.
- Missing local data: Common Voice, FeruzaSpeech, FLEURS, UzbekVoice, YouTube/news/podcast corpora, calls, meetings.

## Which Datasets To Use

### Must Use

- USC
- Common Voice Uzbek cleaned subset
- FeruzaSpeech
- FLEURS Uzbek

### Use After Filtering

- UzbekVoice filtered
- IT YouTube Uzbek Speech
- News YouTube Uzbek Speech
- Podcasts Tashkent Dialect

### Use Only After Strong Teacher

- UzbekVoice raw
- Saidakmal derivatives
- Self-collected YouTube
- Calls/meetings/telephony

## What To Discard

Discard or quarantine:

- Exact duplicate audio.
- Train/val/test leakage.
- Empty or near-empty transcripts.
- Samples with impossible chars/sec.
- High-silence segments with weak transcript confidence.
- Low teacher similarity and high WER/CER.
- Pseudo-labels with hallucination or repeated phrases.
- License-unclear scraped content until approved.

## Path To 500h

1. Combine gold: USC + CV + Feruza + FLEURS, target ~170-220h.
2. Add filtered UzbekVoice plus highest-quality YouTube/news/podcast slices.
3. Dedup globally and score quality.
4. Train Stage 2 with gold overweighted.

Expected impact:

- Large improvement in real-world robustness.
- Moderate clean-test WER improvement beyond USC-only full FT.

## Path To 1000h

1. Expand silver YouTube/podcast/news sources.
2. Add conversational and dialect collections.
3. Use current best model as teacher for pseudo-label scoring.
4. Human-review disagreement bands.

Expected impact:

- Major improvement for podcasts, meetings, spontaneous speech.
- Lower hallucination and Turkish/Kazakh confusion from broader Uzbek prior.

## Path To 1500h

1. Add calls, meetings, telephony, and self-collected YouTube.
2. Build domain-specific validation sets.
3. Introduce Stage 3 bronze training with strict filtering.
4. Finish with Stage 4 domain adaptation for enterprise target domain.

Expected impact:

- Best path to SOTA production Uzbek ASR.
- Gains depend more on label quality and domain diversity than raw hours.

## Expected WER Improvements

These are planning estimates, not guarantees:

- USC partial FT baseline: 20.05% WER.
- USC full FT layer-wise LR: likely improves Uzbek language prior; final WER unknown until current run finishes.
- Gold 170-220h: expected to improve clean Uzbek and reduce script normalization errors.
- 500h mixed gold/silver: expected biggest production robustness jump.
- 1000-1500h with calls/meetings: required for enterprise-grade meetings/calls/podcasts.

## Critical Assumptions To Challenge

1. More data is not always better; low-quality pseudo-labels can degrade WER.
2. Silver and bronze must not dominate batches.
3. USC validation is not sufficient for production quality.
4. Tashkent-heavy podcasts do not solve dialect coverage.
5. Teacher filtering can preserve teacher bias; include human-reviewed disagreement slices.

## Immediate Next Experiment After Current Run

Build a gold-only combined corpus and run Stage 1:

- USC + Common Voice + Feruza + FLEURS.
- Dedup before split.
- Normalize all text.
- Weighted sampling still enabled, but all gold.
- Evaluate on USC test plus a new mixed gold validation set.

