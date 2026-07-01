# Project Charter

Last cleaned for obsolete documentation sections: `2026-07-01T04:52:10Z`.

## Mission

Build the best possible open-weight Uzbek ASR system using `openai/whisper-large-v3`. Uzbek WER/CER is the only optimization target. Multilingual retention is not required.

## Success Metrics

- Primary: validation/test WER and CER on locked Uzbek evaluation sets.
- Secondary: hallucination rate and language-confusion rate.
- Operational: resumable training, preserved best checkpoints, no test leakage, no disk-full crashes.

## Current Direction

The current direction is Stage 1 Gold+Silver training with encoder 0-7 frozen and encoder 8-31 plus decoder trained at 2e-5.
