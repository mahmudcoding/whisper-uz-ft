# Decision Log

**Document role:** Durable record of consequential technical decisions and rationale.  
**Format:** Each entry states decision, evidence, impact, risk, and reversal condition.

## D001 - Optimize Uzbek Only

**Decision:** Minimize Uzbek WER/CER even if other languages degrade.

**Evidence:** Raw multilingual priors produce Turkish/Kazakh-like output and
hallucinations on Uzbek.

**Impact:** Allows aggressive decoder adaptation and Uzbek-specific normalization.

**Risk:** Non-Uzbek transcription quality will decline.

**Revisit when:** A multilingual requirement becomes part of the product.

## D002 - Force Uzbek Transcription

**Decision:** Always set `language="uz"` and `task="transcribe"`.

**Evidence:** Automatic language selection is a known source of Uzbek confusion.

**Impact:** More stable Uzbek decoder prior.

**Risk:** Non-Uzbek audio may be forced into Uzbek-like text.

**Revisit when:** A separate language-routing layer is introduced.

## D003 - Protect the Partial-FT Baseline

**Decision:** Treat `models/partial_ft_usc_baseline/` as immutable.

**Evidence:** It remains the best completed model at WER 20.05%, CER 5.29%.

**Impact:** Guarantees a stable promotion reference and recovery artifact.

**Risk:** Storage cost.

**Revisit when:** A superior model is independently archived and verified; never delete
the historical baseline solely because it is no longer best.

## D004 - Do Not Default to Full FT

**Decision:** Search decoder and upper-encoder adaptation before another full FT.

**Evidence:** USC full FT produced WER 22.22% versus partial FT 20.05%.

**Impact:** Preserves transferable lower-encoder acoustics and reduces overfitting risk.

**Risk:** Frozen lower layers may eventually limit domain adaptation.

**Revisit when:** A much larger/diverse corpus shows consistent gains from deeper
unfreezing.

## D005 - Use Layer-Wise LR

**Decision:** Separate encoder and decoder learning rates.

**Evidence:** Decoder language prior appears to be the dominant Uzbek failure mode.

**Impact:** Enables aggressive linguistic adaptation with conservative acoustic
updates.

**Risk:** Poor LR ratios can destabilize the decoder or under-adapt acoustics.

**Revisit when:** LR/freeze search provides measured alternatives.

## D006 - Use BF16 on A40

**Decision:** Use BF16 for new large-v3 training.

**Evidence:** A40 reports BF16 support; 100-step full-FT dry run completed stably.

**Impact:** Avoids FP16 scaling limitations while fitting large-v3.

**Risk:** Kernel/library-specific numerical behavior.

**Revisit when:** Measured FP16 or another precision is faster and equally stable.

## D007 - One Epoch Before Multi-Epoch Training

**Decision:** Start new expensive corpus/regime experiments with one epoch.

**Evidence:** The initial four-epoch launch was premature; USC full FT underperformed
after one epoch.

**Impact:** Generates evidence before multi-day commitment.

**Risk:** Underfitting.

**Revisit when:** Validation is still improving materially at epoch end and overfitting
signals are absent.

## D008 - Validation-Only Search

**Decision:** LR-search jobs neither load nor evaluate test manifests.

**Evidence:** Loading test during search creates unnecessary leakage risk.

**Impact:** Strong benchmark integrity. Enforced by config validation, leakage audit,
hash guards, and comparison-tool rejection of test metrics.

**Risk:** Fewer observations during search.

**Revisit when:** Never for hyperparameter selection; final test remains post-lock only.

## D009 - Promote Top Two Decoder LRs

**Decision:** Phase 1B promotes two candidates, not only the numerical winner.

**Evidence:** A 1h proxy has sampling variance; tiny WER differences can be noise.

**Impact:** Reduces overfitting to coarse proxy noise.

**Risk:** Additional compute.

**Tie thresholds:** WER `0.003`, CER `0.001`.

## D010 - Build Gold Before Silver/Bronze

**Decision:** Establish a clean 207h Gold corpus before large noisy-data expansion.

**Evidence:** RubaiSTT analysis indicates data scale/diversity and normalization are
major quality drivers, but noisy data can harm training without a strong teacher.

**Impact:** Trusted foundation for teacher and curriculum stages.

**Risk:** Slower path to raw hour growth.

**Revisit when:** Gold model is strong enough to pseudo-label and filter reliably.

## D011 - Canonical Uzbek Latin

**Decision:** Normalize Uzbek Cyrillic/mixed script to canonical Latin.

**Evidence:** Script variation inflates WER/CER and fragments decoder targets.

**Impact:** Consistent supervision and evaluation.

**Risk:** Transliteration ambiguity and possible loss of intentional Cyrillic form.

**Revisit when:** Product requirements demand script-preserving output.

## D012 - Trust-Weighted Curriculum

**Decision:** Gold must be oversampled relative to Silver/Bronze.

**Initial weights:** Gold 4.0, Silver 1.5, Bronze 1.0.

**Evidence:** Large noisy corpora can drown clean supervision.

**Impact:** Preserves label quality while adding domain diversity.

**Risk:** Weight choice may underuse useful Silver data.

**Revisit when:** Curriculum ablations provide measured optimal ratios.

## D013 - Use Measured Long-Form Capacity

**Decision:** Base offline capacity planning on 5h long-form benchmarks, not smoke.

**Evidence:** Smoke and long-form throughput differed materially.

**Impact:** More realistic cost and GPU estimates.

**Risk:** Still limited to A40, beam 1, and the tested workload.

**Revisit when:** Winning checkpoint and additional hardware are measured.

## D014 - Use Kotib as the Silver Filtering Teacher

**Decision:** Use pinned `Kotib/uzbek_stt_v1` revision
`0e239511f65c1c7bbf426619a1ee9ea628411344` for Silver transcript agreement.
Force `language=uz` and `task=transcribe`; do not use automatic language detection
as a rejection signal.

**Evidence:** The USC-only partial model mislabeled accurate Uzbek speech as English,
Persian, Kazakh, and other languages. All first 3,621 candidates were rejected.
Kotib exactly transcribed the eight reproduced failures with zero normalized WER/CER.

**Impact:** Silver filtering uses a stronger Uzbek-only model trained on substantially
more diverse data. Agreement scoring ignores punctuation while retaining punctuation
in the training labels.

**Risk:** Kotib may have trained on overlapping public datasets, making agreement less
independent for those sources. Audio/Gold deduplication and strict heuristic gates
remain mandatory; teacher agreement is evidence, not ground truth.

**Revisit when:** A stronger independently evaluated Uzbek teacher is available.

## D015 - Integrate Feruza with a 30-Second Hard Limit

**Decision:** Add prepared FeruzaSpeech to Gold while excluding clips longer than
30 seconds when transcript-aligned chunk timestamps are unavailable.

**Evidence:** Whisper feature extraction uses a 30-second window. Training against a
full transcript after silent audio truncation creates incorrect supervision. Feruza
provided 136 such clips; retaining the other clips added 57.8279 validated hours.

**Impact:** Gold increases from 207.1150h to 264.9430h without introducing known
path, content-hash, or reliable-speaker leakage.

**Risk:** The gated K2Speech terms restrict use to academic research/internal use and
prohibit redistribution. This dataset must not be assumed suitable for commercial
training or model publication without rights review.

**Revisit when:** Aligned timestamps permit safe chunking, or written commercial and
redistribution rights are obtained.

## D016 - Retain Metrics, Not Search Checkpoints

**Decision:** Keep promoted model weights and active resume state, but delete completed
LR-search checkpoints/final models after metrics and plots are written. Do not keep
timestamped local backups or archive directories.

**Evidence:** Completed checkpoints, duplicate models, dataset caches, and source
archives consumed more than 1 TB while model selection depends on compact metrics.

**Impact:** Project storage fell from approximately 646 GB to 15 GB, and filesystem
free space increased to approximately 1.5 TB. Future LR runs clean their heavyweight
artifacts automatically after successful metric collection.

**Risk:** A completed search model cannot be resumed or inspected at weight level.
Selected hyperparameters remain reproducible from configs, metrics, logs, and the base
model.

**Revisit when:** A candidate is promoted; copy that model into `models/` before
cleanup or set `retain_search_artifacts: true`.
