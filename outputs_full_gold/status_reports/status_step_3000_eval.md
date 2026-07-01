# Full FT Status: step 3000

- Type: `eval`
- Timestamp UTC: `2026-06-28T23:06:34Z`
- Progress: `55.762081784386616`
- ETA seconds: `18854.081942067147`
- Train loss: `1.452566909790039`
- Eval loss: `0.1950564682483673`
- Eval WER: `0.18722939501501737`
- Eval CER: `0.0493969633199828`
- Eval hallucination rate: `0.0013183915622940012`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 16663.30712890625, 'torch_reserved_mib': 37684.0, 'torch_peak_allocated_mib': 35786.62353515625, 'gpu_util_percent': 28.0, 'vram_used_mib': 38029.0, 'vram_total_mib': 46068.0, 'power_watts': 156.87, 'temperature_c': 57.0}`

## Sample Predictions
### Sample 0
- Prediction: `xola, anu shokoladingizdan obering, dedi chamasi yetti yasharlar qiz bildirishlab`
- Reference: `xola, anuv shokoladingizdan obering, dedi chamasi etti yasharlar qiz bidirlab`
- Indicators: `{'prediction_length': 81, 'reference_length': 77, 'length_ratio': 1.051948051948052, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `men bu kunni hayotimdagi eng baxtli kun deb hisoblayman.`
- Reference: `men bu kunni hayotimdagi eng baxtli kun deb hisoblayman.`
- Indicators: `{'prediction_length': 56, 'reference_length': 56, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `taziyq va zo'ravonlikka qarshimiz.`
- Reference: `tazyiq va zo'ravonlikka qarshimiz`
- Indicators: `{'prediction_length': 34, 'reference_length': 33, 'length_ratio': 1.0303030303030303, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
