# coarse_10h LR Search Subset

- Deterministic seed: `1729`
- Training hours: `9.9971`
- Training samples: `8733`
- Training speakers: `629`
- Average duration: `4.121s`
- Duration range: `1.000s` to `27.420s`

## Training Composition

| Dataset | Hours | Share | Samples | Target share |
|---|---:|---:|---:|---:|
| usc | 4.9994 | 50.01% | 5097 | 50% |
| common_voice_uz | 3.9991 | 40.00% | 3340 | 40% |
| fleurs_uz | 0.9987 | 9.99% | 296 | 10% |

## Evaluation Manifests

- `val.csv`: 0.9988h, 845 samples, 338 recorded speaker IDs
- `test.csv`: 0.9992h, 818 samples, 325 recorded speaker IDs

## Sampling Method

Rows are selected only from the matching Gold split. Within each source, sampling is
stratified jointly by duration quintile and transcript-word-count quartile. Stable
SHA-256 ordering makes selection reproducible and independent of input row order.
The validator enforces path separation and reliable speaker separation across splits.
