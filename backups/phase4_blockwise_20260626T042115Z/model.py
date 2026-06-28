from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    tuning_mode: str | None = None,
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

    mode_to_blocks: dict[str, int | str] = {
        "decoder_only": 0,
        "encoder_24_31_plus_decoder": 8,
        "encoder_16_31_plus_decoder": 16,
        "full": "all",
        "full_ft": "all",
    }
    if tuning_mode:
        if tuning_mode not in mode_to_blocks:
            raise ValueError(f"Unsupported tuning_mode={tuning_mode!r}; expected one of {sorted(mode_to_blocks)}")
        train_last_encoder_blocks = mode_to_blocks[tuning_mode]

    full_ft = train_last_encoder_blocks in (None, "all", "full", -1)
    if full_ft:
        for param in model.parameters():
            param.requires_grad = True
    else:
        train_last_encoder_blocks = int(train_last_encoder_blocks)
        if not 0 <= train_last_encoder_blocks <= len(model.model.encoder.layers):
            raise ValueError(
                f"train_last_encoder_blocks must be between 0 and {len(model.model.encoder.layers)}, "
                f"got {train_last_encoder_blocks}"
            )
        for param in model.parameters():
            param.requires_grad = False
        train_last_encoder_blocks = int(train_last_encoder_blocks)

        encoder_layers = model.model.encoder.layers
        if train_last_encoder_blocks > 0:
            start = len(encoder_layers) - train_last_encoder_blocks
            for layer in encoder_layers[start:]:
                for param in layer.parameters():
                    param.requires_grad = True

        for param in model.model.decoder.parameters():
            param.requires_grad = True
        for param in model.proj_out.parameters():
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


def detailed_trainable_parameter_report(model: WhisperForConditionalGeneration) -> dict[str, Any]:
    def summarize(parameters) -> dict[str, Any]:
        params = []
        seen: set[int] = set()
        for param in parameters:
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
        total = sum(param.numel() for param in params)
        trainable = sum(param.numel() for param in params if param.requires_grad)
        states = {param.requires_grad for param in params}
        state = "trainable" if states == {True} else "frozen" if states == {False} else "mixed"
        return {
            "state": state,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
        }

    encoder_layers = model.model.encoder.layers
    report = {
        "encoder_0_7": summarize(param for layer in encoder_layers[0:8] for param in layer.parameters()),
        "encoder_8_23": summarize(param for layer in encoder_layers[8:24] for param in layer.parameters()),
        "encoder_24_31": summarize(param for layer in encoder_layers[24:32] for param in layer.parameters()),
        "decoder": summarize(
            list(model.model.decoder.parameters()) + list(model.proj_out.parameters())
        ),
        "overall": trainable_parameter_report(model),
    }
    report["overall"]["frozen_parameters"] = (
        report["overall"]["total_parameters"] - report["overall"]["trainable_parameters"]
    )
    return report
