# Final LR Search Recommendation

## Selected Configuration

- Best decoder LR: `8e-06`
- Best upper encoder LR: `5e-06`
- Best freeze boundary: `encoder_16_31_plus_decoder`
- Best regime: **C) encoder 16-31 + decoder**
- Validation WER: `0.23663208300079808`
- Validation CER: `0.060960807056348286`
- Confidence: **medium**

## Regime Conclusion

- A) decoder only: `not selected`
- B) encoder 24-31 + decoder: `not selected`
- C) encoder 16-31 + decoder: `SELECTED`
- D) full FT: **rejected for current data scale**; the measured USC full-FT run
  degraded WER/CER relative to partial FT and provides no evidence for lower-encoder updates.

## Recommended 207h Gold Training Config

```yaml
model_name: openai/whisper-large-v3
language: uz
task: transcribe
data_dir: ~/whisper-uz-ft/data/gold_master_training_schema
tuning_mode: encoder_16_31_plus_decoder
decoder_learning_rate: 8e-06
encoder_learning_rate: 5e-06
learning_rate: 8e-06
epochs: 1
per_device_batch_size: 1
gradient_accumulation_steps: 32
warmup_ratio: 0.10
weight_decay: 0.01
max_grad_norm: 1.0
scheduler: cosine
bf16: true
fp16: false
gradient_checkpointing: true
metric_for_best_model: wer
greater_is_better: false
generation_num_beams: 1
```

The final Gold run should preserve a locked test set and use validation-only checkpoint
selection. One epoch is the initial production recommendation; extend only if validation
continues improving without overfitting.
