# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-25T00:23:21Z`
- Progress: `73.0816077953715`
- ETA seconds: `5840.899702525934`
- Train loss: `8.928568267822266`
- Eval loss: `0.35289570689201355`
- Eval WER: `0.3280127693535515`
- Eval CER: `0.08647802983303213`
- Eval hallucination rate: `0.0`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 14329.52587890625, 'torch_reserved_mib': 22826.0, 'torch_peak_allocated_mib': 21974.939453125, 'gpu_util_percent': 46.0, 'vram_used_mib': 23171.0, 'vram_total_mib': 46068.0, 'power_watts': 151.37, 'temperature_c': 53.0}`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassasalarida ushbu kasalliklaridan barvart aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 117, 'reference_length': 115, 'length_ratio': 1.017391304347826, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `briyant o'zini qora manba deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 36, 'reference_length': 36, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `biryan o'zining sermaqsuliga bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 42, 'reference_length': 44, 'length_ratio': 0.9545454545454546, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
