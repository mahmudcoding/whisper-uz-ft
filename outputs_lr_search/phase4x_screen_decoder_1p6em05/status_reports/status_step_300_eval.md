# Full FT Status: step 300

- Type: `eval`
- Timestamp UTC: `2026-06-26T23:33:10Z`
- Progress: `100.0`
- ETA seconds: `0.0`
- Train loss: `12.219969940185546`
- Eval loss: `0.4952670931816101`
- Eval WER: `0.45201919232307075`
- Eval CER: `0.14294445920807866`
- Eval hallucination rate: `0.002366863905325444`
- Eval language confusion rate: `0.001183431952662722`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13125.3125, 'torch_reserved_mib': 21138.0, 'torch_peak_allocated_mib': 20170.283203125, 'gpu_util_percent': 25.0, 'vram_used_mib': 21483.0, 'vram_total_mib': 46068.0, 'power_watts': 151.99, 'temperature_c': 52.0}`

## Sample Predictions
### Sample 0
- Prediction: `fransiya va yaponiya uxani shahridan o'z qoralarini olib chiqib ketayotganini ma'lum qildi`
- Reference: `fransiya va yaponiya uxan shahridan o'z fuqarolarini olib chiqib ketayotganini ma'lum qildi`
- Indicators: `{'prediction_length': 90, 'reference_length': 91, 'length_ratio': 0.989010989010989, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `axoli o'rtasida mavsumi yuqimni kasallikka chalinganning barchasi nazoratga olingan va tegishli dava choralari amalga oshirilmoqda`
- Reference: `aholi o'rtasida mavsumiy yuqumli kasallikka chalinganlarning barchasi nazoratga olingan va tegishli davo choralari amalga oshirilmoqda`
- Indicators: `{'prediction_length': 130, 'reference_length': 134, 'length_ratio': 0.9701492537313433, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `yoshda gidan los angeles lakers ishqib ozib bo'lgan`
- Reference: `yoshligidan los anjeles leykers ishqibozi bo'lgan`
- Indicators: `{'prediction_length': 51, 'reference_length': 49, 'length_ratio': 1.0408163265306123, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
