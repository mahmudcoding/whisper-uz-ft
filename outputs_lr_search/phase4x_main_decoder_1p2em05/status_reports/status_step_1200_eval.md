# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-26T22:12:25Z`
- Progress: `73.0816077953715`
- ETA seconds: `5707.486523448626`
- Train loss: `7.917946624755859`
- Eval loss: `0.34398654103279114`
- Eval WER: `0.5632482043096568`
- Eval CER: `0.2918026596175114`
- Eval hallucination rate: `0.021251475796930343`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13125.31298828125, 'torch_reserved_mib': 21142.0, 'torch_peak_allocated_mib': 20175.556640625, 'gpu_util_percent': 47.0, 'vram_used_mib': 21487.0, 'vram_total_mib': 46068.0, 'power_watts': 153.43, 'temperature_c': 54.0}`
- Stop reason: `eval_wer worsened for two consecutive evaluations: [0.44014365522745413, 0.5440941739824421, 0.5632482043096568]`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassasalarida ushbu kasalliklardan barvaq aniqlash ularni alohida davolash uchun barcha sharoatlari mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 116, 'reference_length': 115, 'length_ratio': 1.008695652173913, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `briyant o'zini qora man bah deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 38, 'reference_length': 36, 'length_ratio': 1.0555555555555556, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `bir yon o'zining ser maqsulligi bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 45, 'reference_length': 44, 'length_ratio': 1.0227272727272727, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
