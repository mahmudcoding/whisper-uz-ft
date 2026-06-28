# Gold Corpus Report

- Final rows: 196,994
- Final hours: 264.9430
- Missing audio: 0
- Cross-split path leakage: 0
- Cross-split audio-hash leakage: 0
- Known speaker leakage: 0

| Dataset | Rows | Hours | Known speakers |
|---|---:|---:|---:|
| common_voice_uz | 72,917 | 88.5389 | 1,366 |
| feruzaspeech | 12,854 | 57.8279 | 127 |
| fleurs_uz | 4,168 | 14.0828 | 2 |
| usc | 107,055 | 104.4933 | 936 |

FeruzaSpeech uses its official speaker-disjoint train/dev/test split. Clips over 30 seconds were excluded before integration because aligned chunk timestamps are unavailable.

FeruzaSpeech licensing is restrictive: academic research/internal use only under the gated K2Speech terms. Do not redistribute the source or prepared audio, and review rights before commercial deployment.
