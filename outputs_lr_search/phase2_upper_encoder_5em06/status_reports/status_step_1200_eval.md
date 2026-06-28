# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-25T20:36:58Z`
- Progress: `73.0816077953715`
- ETA seconds: `7362.059048069319`
- Train loss: `7.339277648925782`
- Eval loss: `0.30807700753211975`
- Eval WER: `0.2789305666400638`
- Eval CER: `0.07560758503493108`
- Eval hallucination rate: `0.0`
- Eval language confusion rate: `0.0011806375442739079`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 14329.52587890625, 'torch_reserved_mib': 22826.0, 'torch_peak_allocated_mib': 21974.939453125, 'gpu_util_percent': 69.0, 'vram_used_mib': 25536.0, 'vram_total_mib': 46068.0, 'power_watts': 212.28, 'temperature_c': 59.0}`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassasalarida ushbu kasalliklaridan barbatov aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 118, 'reference_length': 115, 'length_ratio': 1.0260869565217392, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `briand o'zini qora manba deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 35, 'reference_length': 36, 'length_ratio': 0.9722222222222222, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `briyan o'zining sermaqsulligi bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 43, 'reference_length': 44, 'length_ratio': 0.9772727272727273, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
