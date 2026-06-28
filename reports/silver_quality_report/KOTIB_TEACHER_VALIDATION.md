# Kotib Teacher Validation

Validated on 2026-06-25 before restarting full Silver scoring.

## Runtime

- Model: `Kotib/uzbek_stt_v1`
- Revision: `0e239511f65c1c7bbf426619a1ee9ea628411344`
- Architecture: Whisper Medium
- Runtime: faster-whisper / CTranslate2
- Device: CUDA
- Compute type: FP16
- Decoding: forced `language=uz`, `task=transcribe`, beam 1

## Reproduction Set

Eight UzbekVoice samples previously rejected by the USC teacher's automatic
language-identification gate were rescored.

- Samples: 8
- Exact normalized matches: 8
- Mean normalized WER: 0.0
- Mean normalized CER: 0.0
- Forced language returned by runtime: `uz`

Agreement normalization ignores punctuation differences. The canonical training
transcript still preserves normalized punctuation.
