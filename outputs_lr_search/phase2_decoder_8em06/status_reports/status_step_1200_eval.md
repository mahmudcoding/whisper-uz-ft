# Full FT Status: step 1200

- Type: `eval`
- Timestamp UTC: `2026-06-24T18:51:04Z`
- Progress: `73.0816077953715`
- ETA seconds: `3661.9164684657253`
- Train loss: `4.697948837280274`
- Eval loss: `0.3672550618648529`
- Eval WER: `2.233240223463687`
- Eval CER: `1.3246567582877027`
- Eval hallucination rate: `0.1381345926800472`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13121.05908203125, 'torch_reserved_mib': 21286.0, 'torch_peak_allocated_mib': 20139.20849609375, 'gpu_util_percent': 46.0, 'vram_used_mib': 21631.0, 'vram_total_mib': 46068.0, 'power_watts': 154.58, 'temperature_c': 50.0}`
- Stop reason: `hallucination rate increased substantially: previous=0.07083825265643448, current=0.1381345926800472`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muassaslarida ushbu kasallarni barbat aniqalash uchun barcha sharoatlari mavjud`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 88, 'reference_length': 115, 'length_ratio': 0.7652173913043478, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `briyant o'zini qora man bah deb atardi`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 38, 'reference_length': 36, 'length_ratio': 1.0555555555555556, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `bir yon o'zining ser maqsulliga bilan mashhur`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 45, 'reference_length': 44, 'length_ratio': 1.0227272727272727, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
