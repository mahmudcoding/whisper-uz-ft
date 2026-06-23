# Data Scaling Plan

Generated: 2026-06-23 UTC

## Goal

Scale Uzbek ASR training data from 104.63h to 500-1500h while improving domain diversity and label quality.

## Target Domains

| Domain | Why It Matters |
| --- | --- |
| Podcasts | long-form spontaneous speech, multiple speakers |
| Interviews | conversational Uzbek, interruptions |
| TV/news | clear speech, named entities |
| Lectures | long recordings, technical terms |
| Meetings | enterprise target domain |
| Calls | channel noise, short turns |
| Webinars | mixed scripted/spontaneous speech |
| Noisy public audio | robustness |

## Pipeline

1. Collect
   - Gather audio with explicit license or internal permission.
   - Store source URL, license, speaker/domain metadata, and collection date.

2. Segment
   - Normalize audio to 16 kHz mono WAV.
   - Use VAD to split into 5-30 second segments.
   - Preserve long-recording IDs for leakage control.

3. Pseudo-label
   - Use current `outputs/final_model` and a stronger teacher if available.
   - Keep raw hypotheses, normalized hypotheses, confidence proxies, and decoding settings.

4. Normalize
   - Apply `src/text_normalization/uz_normalizer.py`.
   - Preserve original text and normalized text.

5. Filter
   - Use CER/WER/similarity scoring.
   - Reject audio/text mismatch, empty speech, extreme chars/sec, and corrupted files.
   - Review dialect/code-switching samples before rejection.

6. Train
   - Start with curriculum: USC + high-confidence pseudo-labeled clean data.
   - Add noisy/conversational domains with controlled sampling.
   - Track per-domain WER, not only aggregate WER.

7. Evaluate
   - Hold out a fixed enterprise benchmark before scaling.
   - Report raw and normalized WER/CER by domain.

## Recommended Milestones

| Milestone | Data Size | Experiment |
| --- | ---: | --- |
| M1 | 104h | Current USC baseline complete |
| M2 | 150-200h | Add high-confidence podcasts/news |
| M3 | 300-500h | Add meetings/calls/webinars |
| M4 | 500-1000h | Full diversity retraining |
| M5 | 1000-1500h | Distillation and production optimization |

## Highest Impact Recommendation

The next model-quality jump should come from data, not more epochs on USC. Add 100-300h of diverse, filtered, normalized audio before investing heavily in long multi-epoch training.

