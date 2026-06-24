# Full FT Status: step 300

- Type: `eval`
- Timestamp UTC: `2026-06-24T07:40:41Z`
- Progress: `100.0`
- ETA seconds: `0.0`
- Train loss: `44.61610717773438`
- Eval loss: `1.427962303161621`
- Eval WER: `0.6445421831267493`
- Eval CER: `0.1610948711134733`
- Eval hallucination rate: `0.0`
- Eval language confusion rate: `0.008284023668639054`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13125.3125, 'torch_reserved_mib': 21126.0, 'torch_peak_allocated_mib': 20170.283203125, 'gpu_util_percent': 29.0, 'vram_used_mib': 21471.0, 'vram_total_mib': 46068.0, 'power_watts': 161.13, 'temperature_c': 53.0}`

## Sample Predictions
### Sample 0
- Prediction: `fransiya va yaponya uxani shahridan o'z qoralarini olib tushqib ketib etganini malim qildi`
- Reference: `fransiya va yaponiya uxan shahridan o'z fuqarolarini olib chiqib ketayotganini ma'lum qildi`
- Indicators: `{'prediction_length': 90, 'reference_length': 91, 'length_ratio': 0.989010989010989, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `axali o'tasida mausumi yuqumni kasarlikga chalinganlarning barchasa nazoratga olingan va tegishli dava charilari amaliga oshilmoqda.`
- Reference: `aholi o'rtasida mavsumiy yuqumli kasallikka chalinganlarning barchasi nazoratga olingan va tegishli davo choralari amalga oshirilmoqda`
- Indicators: `{'prediction_length': 132, 'reference_length': 134, 'length_ratio': 0.9850746268656716, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `yoshda gidan los ang'lis, lakers ishqib o'zi bo'lgan`
- Reference: `yoshligidan los anjeles leykers ishqibozi bo'lgan`
- Indicators: `{'prediction_length': 52, 'reference_length': 49, 'length_ratio': 1.0612244897959184, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
