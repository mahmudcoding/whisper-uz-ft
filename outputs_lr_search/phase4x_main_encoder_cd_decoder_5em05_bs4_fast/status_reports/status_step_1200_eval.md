# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-27T15:09:50Z`
- Progress: `73.0816077953715`
- ETA seconds: `2200.046342774232`
- Train loss: `1.1878959655761718`
- Eval loss: `0.2494397610425949`
- Eval WER: `0.21428571428571427`
- Eval CER: `0.05758908100234672`
- Eval hallucination rate: `0.0023612750885478157`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 15473.37158203125, 'torch_reserved_mib': 32790.0, 'torch_peak_allocated_mib': 31377.78173828125, 'gpu_util_percent': 45.0, 'vram_used_mib': 33135.0, 'vram_total_mib': 46068.0, 'power_watts': 158.49, 'temperature_c': 55.0}`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash ish uchun barcha sharoitlar mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 119, 'reference_length': 115, 'length_ratio': 1.0347826086956522, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `brayant o'zini qora manba deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 36, 'reference_length': 36, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `briyan o'tining sermahsuligi bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 42, 'reference_length': 44, 'length_ratio': 0.9545454545454546, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
