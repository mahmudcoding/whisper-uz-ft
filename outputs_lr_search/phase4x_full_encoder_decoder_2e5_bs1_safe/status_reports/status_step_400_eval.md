# Full FT Status: step 400

- Type: `eval`
- Timestamp UTC: `2026-06-28T01:00:43Z`
- Progress: `24.3605359317905`
- ETA seconds: `18139.20409434557`
- Train loss: `12.625533294677734`
- Eval loss: `0.3647635281085968`
- Eval WER: `3.9112130885873904`
- Eval CER: `2.2549564372993824`
- Eval hallucination rate: `0.5608028335301063`
- Eval language confusion rate: `0.0`
- GPU: `{'cuda_available': True, 'torch_allocated_mib': 18028.462890625, 'torch_reserved_mib': 32336.0, 'torch_peak_allocated_mib': 30554.32861328125, 'gpu_util_percent': 44.0, 'vram_used_mib': 32681.0, 'vram_total_mib': 46068.0, 'power_watts': 127.45, 'temperature_c': 49.0}`
- Stop reason: `hallucination rate increased substantially: previous=0.0, current=0.5608028335301063`

## Sample Predictions
### Sample 0
- Prediction: `tibbiyot muvot muvotatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatatat`
- Reference: `tibbiyot muassasalarida ushbu kasalliklarni barvaqt aniqlash ularni alohida davolash uchun barcha sharoitlar mavjud`
- Indicators: `{'prediction_length': 120, 'reference_length': 115, 'length_ratio': 1.0434782608695652, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': False}`
### Sample 1
- Prediction: `bryant o'z o'z o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o`
- Reference: `brayant o'zini qora mamba deb atardi`
- Indicators: `{'prediction_length': 98, 'reference_length': 36, 'length_ratio': 2.7222222222222223, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': True}`
### Sample 2
- Prediction: `bir yand o'z o'z o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o o`
- Reference: `brayant o'zining sermahsulligi bilan mashhur`
- Indicators: `{'prediction_length': 94, 'reference_length': 44, 'length_ratio': 2.1363636363636362, 'language_confusion': False, 'unexpected_chars': [], 'hallucination': True}`
