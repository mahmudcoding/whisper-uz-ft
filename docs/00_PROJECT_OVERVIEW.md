# Project Overview

## Mission

Build the best open-weight Uzbek ASR system using `openai/whisper-large-v3` as the base model.

The optimization target is Uzbek recognition quality only:

- Primary metric: Uzbek WER.
- Secondary metric: Uzbek CER.
- Catastrophic forgetting of non-Uzbek languages is acceptable.
- Preserving English, Russian, Turkish, or general multilingual Whisper behavior is not a goal for this model.

## Production Use Cases

The target deployment is high-quality offline enterprise transcription for:

- Meetings.
- Calls.
- Webinars.
- Podcasts.
- Noisy real-world Uzbek speech.
- Uzbek Latin preferred output.
- Mixed-script Uzbek support.
- Uzbek-Russian and Uzbek-English code-switching where present in the audio.

## Current Technical Direction

The project moved from partial fine-tuning to aggressive Uzbek-only full fine-tuning.

Key strategic decisions:

- Force Whisper decoding to Uzbek: `language="uz"`, `task="transcribe"`.
- Fine-tune all 1,543,490,560 Whisper large-v3 parameters for Uzbek-only adaptation.
- Use BF16 on NVIDIA A40 because local PyTorch reports BF16 support.
- Use layer-wise learning rates to adapt decoder language prior faster than encoder acoustics:
  - Encoder LR: `2e-6`.
  - Decoder LR: `8e-6`.
- Normalize all transcripts to canonical Uzbek Latin before serious training and evaluation.
- Build a trust-tiered corpus and prevent Gold data from being drowned by noisier data.

## Success Criteria

Minimum acceptable near-term target:

- Beat the archived partial fine-tune USC baseline:
  - WER: `20.05%`.
  - CER: `5.29%`.

Longer-term target:

- Build a 500-1500h filtered Uzbek corpus.
- Improve robustness on calls, meetings, podcasts, news, dialect speech, noisy audio, and code-switching.
- Establish a reproducible evaluation suite comparing raw Whisper, internal checkpoints, RubaiSTT, and Kotib baselines.

## Repository Root

Project root:

```bash
/home/mahmud/whisper-uz-ft
```

External staged dataset root:

```bash
/home/mahmud/datasets
```

## Documentation Authority

The files in `docs/00_*` through `docs/15_*` are the authoritative project memory. Historical docs are archived under `docs/archive/` and must not be treated as current truth unless explicitly referenced by the numbered docs.
