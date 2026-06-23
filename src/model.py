from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass
class ModelBundle:
    model: WhisperForConditionalGeneration
    processor: WhisperProcessor


def load_whisper_for_partial_ft(
    model_name: str = "openai/whisper-large-v3",
    language: str = "uz",
    task: str = "transcribe",
    train_last_encoder_blocks: int | str | None = 8,
    gradient_checkpointing: bool = True,
) -> ModelBundle:
    if language != "uz" or task != "transcribe":
        raise ValueError("This project is configured for Uzbek-only ASR: language='uz', task='transcribe'.")

    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(dtype=torch.float32)

    model.config.use_cache = False
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.generation_config.suppress_tokens = []

    full_ft = train_last_encoder_blocks in (None, "all", "full", -1)
    if full_ft:
        for param in model.parameters():
            param.requires_grad = True
    else:
        train_last_encoder_blocks = int(train_last_encoder_blocks)
        for param in model.model.encoder.parameters():
            param.requires_grad = False

        encoder_layers = model.model.encoder.layers
        start = max(0, len(encoder_layers) - train_last_encoder_blocks)
        for layer in encoder_layers[start:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in model.model.decoder.parameters():
            param.requires_grad = True

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return ModelBundle(model=model, processor=processor)


def trainable_parameter_report(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float(trainable / total * 100.0) if total else 0.0,
    }
