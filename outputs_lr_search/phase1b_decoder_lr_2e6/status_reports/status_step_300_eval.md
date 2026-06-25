# Full FT Status: step 300

- Type: `eval`
- Timestamp UTC: `2026-06-24T12:04:15Z`
- Progress: `54.94505494505495`
- ETA seconds: `2289.044690699577`
- Train loss: `15.05301513671875`
- Eval loss: `0.959773063659668`
- Eval WER: `0.6103558576569372`
- Eval CER: `0.1718044113739038`
- Eval hallucination rate: `0.001183431952662722`
- Eval language confusion rate: `0.007100591715976331`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13121.74072265625, 'torch_reserved_mib': 21276.0, 'torch_peak_allocated_mib': 20140.8525390625, 'gpu_util_percent': 44.0, 'vram_used_mib': 21621.0, 'vram_total_mib': 46068.0, 'power_watts': 145.14, 'temperature_c': 57.0}`

## Sample Predictions
### Sample 0
- Prediction: `fransiya va yaponya uxana shahridan o'z qoralarini olib tushqib ketib etganini ma'lum qildi`
- Reference: `fransiya va yaponiya uxan shahridan o'z fuqarolarini olib chiqib ketayotganini ma'lum qildi`
- Indicators: `{'prediction_length': 91, 'reference_length': 91, 'length_ratio': 1.0, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `axali o'tasida mausumi yuqumni kasarlikga chalinganlarning barchasa nazoratga olingan va tegishli davacharalar amalga oshilmoqda`
- Reference: `aholi o'rtasida mavsumiy yuqumli kasallikka chalinganlarning barchasi nazoratga olingan va tegishli davo choralari amalga oshirilmoqda`
- Indicators: `{'prediction_length': 128, 'reference_length': 134, 'length_ratio': 0.9552238805970149, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `yoshda gidan los ang'lis, lakers ishqib o'zi bo'lgan`
- Reference: `yoshligidan los anjeles leykers ishqibozi bo'lgan`
- Indicators: `{'prediction_length': 52, 'reference_length': 49, 'length_ratio': 1.0612244897959184, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
