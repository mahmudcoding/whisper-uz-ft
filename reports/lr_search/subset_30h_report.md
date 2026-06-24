# main_30h LR Search Subset

- Deterministic seed: `1829`
- Training hours: `29.9987`
- Training samples: `26249`
- Training speakers: `805`
- Average duration: `4.114s`
- Duration range: `1.000s` to `28.920s`

## Training Composition

| Dataset | Hours | Share | Samples | Target share |
|---|---:|---:|---:|---:|
| usc | 14.9999 | 50.00% | 15323 | 50% |
| common_voice_uz | 11.9996 | 40.00% | 10040 | 40% |
| fleurs_uz | 2.9993 | 10.00% | 886 | 10% |

## Evaluation Manifests

- `val.csv`: 1.0002h, 847 samples, 328 recorded speaker IDs
- `test.csv`: 1.0002h, 816 samples, 329 recorded speaker IDs

## Sampling Method

Rows are selected only from the matching Gold split. Within each source, sampling is
stratified jointly by duration quintile and transcript-word-count quartile. Stable
SHA-256 ordering makes selection reproducible and independent of input row order.
The validator enforces path separation and reliable speaker separation across splits.
