# Full FT Status: step 5380

- Type: `eval`
- Timestamp UTC: `2026-06-29T04:55:09Z`
- Progress: `100.0`
- ETA seconds: `0.0`
- Train loss: `1.174937629699707`
- Eval loss: `0.15512986481189728`
- Eval WER: `0.14520029644654212`
- Eval CER: `0.03692389694697752`
- Eval hallucination rate: `0.0006591957811470006`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 16626.8955078125, 'torch_reserved_mib': 37684.0, 'torch_peak_allocated_mib': 35789.31201171875, 'gpu_util_percent': 45.0, 'vram_used_mib': 38029.0, 'vram_total_mib': 46068.0, 'power_watts': 143.86, 'temperature_c': 59.0}`

## Sample Predictions
### Sample 0
- Prediction: `- xola, anu shokoladingizdan obering, - dedi chamasi yetti yasharlar qiz bildirishlab.`
- Reference: `xola, anuv shokoladingizdan obering, dedi chamasi etti yasharlar qiz bidirlab`
- Indicators: `{'prediction_length': 86, 'reference_length': 77, 'length_ratio': 1.1168831168831168, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `men bu kunni hayotimdagi eng baxtli kun deb hisoblayman.`
- Reference: `men bu kunni hayotimdagi eng baxtli kun deb hisoblayman.`
- Indicators: `{'prediction_length': 56, 'reference_length': 56, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `tazyiq va zo'ravonlikka qarshimiz.`
- Reference: `tazyiq va zo'ravonlikka qarshimiz`
- Indicators: `{'prediction_length': 34, 'reference_length': 33, 'length_ratio': 1.0303030303030303, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
