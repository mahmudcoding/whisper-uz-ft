# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-28T08:28:48Z`
- Progress: `73.0816077953715`
- ETA seconds: `2187.4804872365794`
- Train loss: `1.1670347213745118`
- Eval loss: `0.24796147644519806`
- Eval WER: `0.21867517956903432`
- Eval CER: `0.06303779030561325`
- Eval hallucination rate: `0.0011806375442739079`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 15473.37158203125, 'torch_reserved_mib': 32790.0, 'torch_peak_allocated_mib': 31377.78173828125, 'gpu_util_percent': 45.0, 'vram_used_mib': 33135.0, 'vram_total_mib': 46068.0, 'power_watts': 155.6, 'temperature_c': 58.0}`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassaslarida ushbu kasalliklardan barvaqt aniqlashish ularni alohida davolashish uchun barcha sharoitlar mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 121, 'reference_length': 115, 'length_ratio': 1.0521739130434782, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `bryant o'zini qora manba deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 35, 'reference_length': 36, 'length_ratio': 0.9722222222222222, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `biron o'zining ser mahsulligi bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 43, 'reference_length': 44, 'length_ratio': 0.9772727272727273, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
