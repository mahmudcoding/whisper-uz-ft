# Full FT Status: step 150

- Type: `eval`
- Timestamp UTC: `2026-06-24T13:11:22Z`
- Progress: `27.472527472527474`
- ETA seconds: `3679.7622820472716`
- Train loss: `12.381938171386718`
- Eval loss: `0.7218828201293945`
- Eval WER: `0.5877648940423831`
- Eval CER: `0.15939410045176722`
- Eval hallucination rate: `0.001183431952662722`
- Eval language confusion rate: `0.0035502958579881655`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13121.74072265625, 'torch_reserved_mib': 21278.0, 'torch_peak_allocated_mib': 20140.8525390625, 'gpu_util_percent': 43.0, 'vram_used_mib': 21623.0, 'vram_total_mib': 46068.0, 'power_watts': 163.65, 'temperature_c': 57.0}`

## Sample Predictions
### Sample 0
- Prediction: `fransiya va yaponiya uxani shahridan o'z qoralarini olib tushqib ketib etganini ma'lum qildi.`
- Reference: `fransiya va yaponiya uxan shahridan o'z fuqarolarini olib chiqib ketayotganini ma'lum qildi`
- Indicators: `{'prediction_length': 93, 'reference_length': 91, 'length_ratio': 1.021978021978022, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `axoli o'rtasida mausumi yuqimni kasallikga chalinganlarning barchasa nazoratga olingan va tegishli dava choralari amaliga oshirilmoqda.`
- Reference: `aholi o'rtasida mavsumiy yuqumli kasallikka chalinganlarning barchasi nazoratga olingan va tegishli davo choralari amalga oshirilmoqda`
- Indicators: `{'prediction_length': 135, 'reference_length': 134, 'length_ratio': 1.007462686567164, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `yoshda gidan los angilis - lakers ishqib ozi bo'lgan`
- Reference: `yoshligidan los anjeles leykers ishqibozi bo'lgan`
- Indicators: `{'prediction_length': 52, 'reference_length': 49, 'length_ratio': 1.0612244897959184, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
