# Dataset Report

Generated: 2026-06-23 UTC

## Corpus

Primary dataset: `issai/Uzbek_Speech_Corpus`

Current local manifests:

| Split | Rows | Hours | Duration Range |
| --- | ---: | ---: | --- |
| train | 99,617 | 96.1401 | 1.0-27.3865 sec |
| val | 3,762 | 3.9967 | 1.0-15.7945 sec |
| test | 3,821 | 4.4930 | 1.0-14.5775 sec |
| total | 107,200 | 104.6298 | 1.0-27.3865 sec |

Mini manifests:

| Split | Rows | Hours |
| --- | ---: | ---: |
| mini_train | 2,085 | 2.0001 |
| mini_val | 375 | 0.3339 |
| mini_test | 279 | 0.3338 |

## Quality Audit

Generated files:

- `reports/dataset_quality_scores.csv`
- `bad_samples.csv`

Current audit results after Uzbek apostrophe normalization:

| Decision | Count |
| --- | ---: |
| keep | 77,539 |
| suspicious | 29,661 |
| reject | 0 |

By split:

| Split | Keep | Suspicious | Reject |
| --- | ---: | ---: | ---: |
| train | 69,993 | 29,624 | 0 |
| val | 3,735 | 27 | 0 |
| test | 3,811 | 10 | 0 |

Top reasons:

| Reason | Count |
| --- | ---: |
| duplicate_transcript | 29,646 |
| high_chars_per_second | 5 |
| duplicate_transcript + low_chars_per_second | 5 |
| low_chars_per_second | 3 |
| duplicate_transcript + high_chars_per_second | 2 |

## Interpretation

The first quality audit was overly conservative because normal Uzbek typographic apostrophes such as `g‘` and `o‘` were counted as suspicious symbols. The scoring pipeline now normalizes text before suspicious-symbol detection.

The remaining suspicious set is dominated by duplicate transcripts. This is not automatically bad: short read-speech corpora often repeat common phrases. It should be used as a review/filtering feature, not as a deletion rule.

## Recommendations

1. Do not remove rows automatically from USC yet.
   - Rationale: there are no automatic rejects after normalization.
   - Expected impact: avoids throwing away valid speech.
   - Risk: noisy duplicates remain until teacher-ASR scoring is added.

2. Add teacher-ASR similarity scoring on a representative subset before data removal.
   - Rationale: transcript/audio mismatch is the most important quality failure.
   - Expected impact: better filtering than text-only heuristics.
   - Risk: teacher model can propagate its own bias.

3. Add domain-diverse data.
   - Rationale: USC is clean read speech and does not cover meetings, calls, webinars, podcasts, or noise well.
   - Expected impact: larger real-world WER reduction than code-level training changes.
   - Risk: pseudo-label quality must be controlled.

## Teacher-ASR Subset

Generated: 2026-06-23 UTC

Generated:

- `reports/teacher_subset_final_model_20.json`

Current final model on first 20 test samples:

- WER: 0.0782
- CER: 0.0139
- RTF: 0.1266
- Peak VRAM: 5465.5 MiB

Observed errors are mostly Uzbek-internal acoustic, suffix, and word-boundary mistakes. No deletion policy should be applied from this small subset alone.
