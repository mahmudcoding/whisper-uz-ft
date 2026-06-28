# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-25T12:57:20Z`
- Progress: `73.0816077953715`
- ETA seconds: `5943.572553505103`
- Train loss: `8.114566040039062`
- Eval loss: `0.328704297542572`
- Eval WER: `0.3026735833998404`
- Eval CER: `0.08027405389366925`
- Eval hallucination rate: `0.0`
- Eval language confusion rate: `0.0011806375442739079`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 14329.52587890625, 'torch_reserved_mib': 22826.0, 'torch_peak_allocated_mib': 21974.939453125, 'gpu_util_percent': 45.0, 'vram_used_mib': 23171.0, 'vram_total_mib': 46068.0, 'power_watts': 159.78, 'temperature_c': 58.0}`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassasalarida ushbu kasalliklardan barvart aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 116, 'reference_length': 115, 'length_ratio': 1.008695652173913, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `briyant o'zini qora manba deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 36, 'reference_length': 36, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `briyan o'zining sermaqsulligi bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 43, 'reference_length': 44, 'length_ratio': 0.9772727272727273, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
