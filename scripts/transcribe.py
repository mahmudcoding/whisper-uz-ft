#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch
from transformers import pipeline


def load_audio_for_pipeline(audio_path: Path, target_sr: int = 16000) -> dict:
    array, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return {"array": array, "sampling_rate": target_sr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with Whisper Uzbek ASR.")
    parser.add_argument("audio_path")
    parser.add_argument("--checkpoint", default=str(Path.home() / "whisper-uz-ft/outputs/final_model"))
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    audio = Path(args.audio_path).expanduser()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        feature_extractor=str(checkpoint),
        torch_dtype=dtype,
        device=device,
    )
    result = asr(
        load_audio_for_pipeline(audio),
        generate_kwargs={"language": "uz", "task": "transcribe", "num_beams": args.beam_size},
    )
    print(result["text"].strip())


if __name__ == "__main__":
    main()
