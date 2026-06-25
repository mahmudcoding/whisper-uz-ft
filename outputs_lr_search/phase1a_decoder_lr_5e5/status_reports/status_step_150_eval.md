# Full FT Status: step 150

- Type: `eval`
- Timestamp UTC: `2026-06-24T11:17:01Z`
- Progress: `50.0`
- ETA seconds: `3345.9887120723724`
- Train loss: `18.362034606933594`
- Eval loss: `0.540521502494812`
- Eval WER: `2.640143942423031`
- Eval CER: `2.14416688812118`
- Eval hallucination rate: `0.2686390532544379`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 13124.4140625, 'torch_reserved_mib': 21142.0, 'torch_peak_allocated_mib': 20170.283203125, 'gpu_util_percent': 46.0, 'vram_used_mib': 21487.0, 'vram_total_mib': 46068.0, 'power_watts': 144.89, 'temperature_c': 55.0}`
- Stop reason: `hallucination rate increased substantially: previous=0.0, current=0.2686390532544379`

## Sample Predictions
### Sample 0
- Prediction: `fransiya va yaponi uxani shahridan o'zfqorini olib chiqib ketganini ma'lum qildi`
- Reference: `fransiya va yaponiya uxan shahridan o'z fuqarolarini olib chiqib ketayotganini ma'lum qildi`
- Indicators: `{'prediction_length': 80, 'reference_length': 91, 'length_ratio': 0.8791208791208791, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `aholi o'rtasida masumi yuqumning masasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiasiga yigitganga davoqarga oshilmoqdan`
- Reference: `aholi o'rtasida mavsumiy yuqumli kasallikka chalinganlarning barchasi nazoratga olingan va tegishli davo choralari amalga oshirilmoqda`
- Indicators: `{'prediction_length': 239, 'reference_length': 134, 'length_ratio': 1.7835820895522387, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 2
- Prediction: `yoshga dengidan los angils lakers ishqib bo'lgan`
- Reference: `yoshligidan los anjeles leykers ishqibozi bo'lgan`
- Indicators: `{'prediction_length': 48, 'reference_length': 49, 'length_ratio': 0.9795918367346939, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
