# Full FT Status: step 2000

- Type: `eval`
- Timestamp UTC: `2026-06-28T20:54:30Z`
- Progress: `37.174721189591075`
- ETA seconds: `26772.81114722967`
- Train loss: `1.8960783004760742`
- Eval loss: `0.2354976087808609`
- Eval WER: `0.21587939306471116`
- Eval CER: `0.05566492430927975`
- Eval hallucination rate: `0.0016479894528675016`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 16663.30712890625, 'torch_reserved_mib': 37684.0, 'torch_peak_allocated_mib': 35786.62353515625, 'gpu_util_percent': 37.0, 'vram_used_mib': 38029.0, 'vram_total_mib': 46068.0, 'power_watts': 136.01, 'temperature_c': 54.0}`

## Sample Predictions
### Sample 0
- Prediction: `xola anu shokoladingizdan obering, dedi chamasi yetti yashorlar qiz bildirishlar`
- Reference: `xola, anuv shokoladingizdan obering, dedi chamasi etti yasharlar qiz bidirlab`
- Indicators: `{'prediction_length': 80, 'reference_length': 77, 'length_ratio': 1.0389610389610389, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `men bu kunni hayotimdagi eng baxtli kun deb hisoblayman.`
- Reference: `men bu kunni hayotimdagi eng baxtli kun deb hisoblayman.`
- Indicators: `{'prediction_length': 56, 'reference_length': 56, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `ta'ziq va zurubonlikka qarshimiz.`
- Reference: `tazyiq va zo'ravonlikka qarshimiz`
- Indicators: `{'prediction_length': 33, 'reference_length': 33, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
