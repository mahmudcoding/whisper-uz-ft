# Full FT Status: step 400

- Type: `eval`
- Timestamp UTC: `2026-06-24T19:58:14Z`
- Progress: `24.3605359317905`
- ETA seconds: `12325.287677096128`
- Train loss: `7.581255340576172`
- Eval loss: `0.4355061948299408`
- Eval WER: `6.313447725458898`
- Eval CER: `2.530062309497478`
- Eval hallucination rate: `0.2526564344746163`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13121.05810546875, 'torch_reserved_mib': 21284.0, 'torch_peak_allocated_mib': 20139.20849609375, 'gpu_util_percent': 33.0, 'vram_used_mib': 21629.0, 'vram_total_mib': 46068.0, 'power_watts': 134.16, 'temperature_c': 49.0}`
- Stop reason: `hallucination rate increased substantially: previous=0.0, current=0.2526564344746163`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muhasalarida ushbu kasalliklaridan barbat aniqa u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u u va tarrarararararar,`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 465, 'reference_length': 115, 'length_ratio': 4.043478260869565, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': True}`
### Sample 1
- Prediction: `briyant o'zini qora ma ma deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 36, 'reference_length': 36, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `biroyan o'zining tex ser maqsulligi bilan mashqur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 49, 'reference_length': 44, 'length_ratio': 1.1136363636363635, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
