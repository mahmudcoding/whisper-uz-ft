# Project Charter

**Document role:** Stable definition of mission, scope, constraints, and success.  
**Change frequency:** Low. Update only when project objectives or constraints change.

## Mission

Build the highest-quality open-weight Uzbek automatic speech recognition system
possible using `openai/whisper-large-v3` as the foundation.

The optimization target is Uzbek transcription quality:

- Primary metric: word error rate (WER).
- Secondary metric: character error rate (CER).
- Operational quality: low hallucination and low language-confusion rates.

Catastrophic forgetting of English, Russian, Turkish, Kazakh, or general multilingual
Whisper behavior is acceptable when it improves Uzbek. Separate models can serve other
languages.

## Intended Product

The model targets offline enterprise transcription for:

- meetings and webinars;
- phone calls and telephony;
- interviews and conversations;
- podcasts, broadcast, and news;
- lectures and long recordings;
- noisy real-world Uzbek;
- mixed Uzbek Latin/Cyrillic input conventions;
- Uzbek speech containing Russian or English code-switching.

Canonical output is Uzbek Latin. The normalizer preserves non-Uzbek text where
possible while standardizing Uzbek script and punctuation.

## Current Technical Thesis

Measured evidence contradicts indiscriminate full fine-tuning on a small clean corpus:

| Regime | Data | WER | CER | Conclusion |
|---|---:|---:|---:|---|
| Raw Whisper large-v3 | USC mini test | 105.22% | 45.90% | Unusable without adaptation |
| Mini adaptation | USC mini | 49.61% | 10.94% | Adaptation is highly effective |
| Partial FT: encoder 24-31 + decoder | USC, 104.63h | **20.05%** | **5.29%** | Current best completed model |
| Full FT with layer-wise LR | USC, 104.63h | 22.22% | 5.66% | Lower encoder adaptation degraded quality |

The working hypothesis is therefore:

1. Whisper's lower encoder contains valuable transferable acoustic representations.
2. Uzbek language-prior errors are concentrated in the decoder and upper encoder.
3. Decoder LR should be optimized first.
4. Upper encoder layers should be unfrozen only when validation evidence supports it.
5. The larger 207.12h Gold corpus should be trained only after proxy LR search selects
   a stable regime.

## Constraints

- Hardware: one NVIDIA A40 48 GB, 52 vCPU, 110 GiB RAM.
- No sudo; all dependencies must be installed in user space.
- Long jobs must survive SSH disconnects.
- Training must be resumable with optimizer, scheduler, and global-step state.
- The protected partial-FT baseline must remain immutable.
- Evaluation integrity takes precedence over speed.
- Large dataset downloads require explicit awareness of licensing, access, and storage.

## Success Criteria

### Near Term

- Complete the LR/freeze-boundary search without test leakage.
- Select decoder LR, upper encoder LR, and freeze boundary from validation evidence.
- Produce a reproducible 207h Gold training configuration.
- Beat the protected partial-FT baseline on a locked test set.

### Medium Term

- Add missing conversational, telephony, meeting, dialect, age, and code-switch data.
- Grow to 500-1500 filtered hours through trust-weighted curriculum training.
- Compare internal models against public Uzbek ASR systems on identical normalized
  evaluation sets.

### Production

- Demonstrate quality on long-form enterprise audio.
- Establish measured A40 inference throughput, memory use, and cost.
- Convert the winning checkpoint to faster-whisper/CTranslate2 without quality loss.
- Maintain a reproducible model registry, evaluation protocol, and disaster-recovery
  path.

## Out of Scope

- Preserving multilingual Whisper quality.
- Training on unfiltered pseudo-labels.
- Using validation or test data as training material.
- Treating smoke benchmarks as production capacity evidence.
- Purchasing hardware based only on modeled, rather than measured, performance.
