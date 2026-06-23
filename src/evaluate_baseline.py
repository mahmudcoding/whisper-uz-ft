from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_audio_array(path: str, target_sr: int = 16000) -> np.ndarray:
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        import librosa

        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return np.asarray(audio, dtype=np.float32)


def load_csv(path: Path):
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    return frame[["audio_path", "text"]].to_dict("records")


def evaluate_split(model, processor, ds: list[dict], batch_size: int, beam_size: int, device: str, dtype: torch.dtype) -> dict:
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    predictions: list[str] = []
    references: list[str] = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="uz", task="transcribe")

    examples = []
    for start in tqdm(range(0, len(ds), batch_size), desc="baseline"):
        batch = ds[start : min(start + batch_size, len(ds))]
        arrays = [load_audio_array(item["audio_path"]) for item in batch]
        inputs = processor.feature_extractor(arrays, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=beam_size,
                language="uz",
                task="transcribe",
            )
        decoded = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded)
        refs = [item["text"] for item in batch]
        references.extend(refs)
        for ref, pred_text in zip(refs, decoded):
            if len(examples) < 20:
                examples.append({"reference": ref, "prediction": pred_text})

    return {
        "wer": float(wer_metric.compute(predictions=predictions, references=references)),
        "cer": float(cer_metric.compute(predictions=predictions, references=references)),
        "samples": len(references),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw Whisper large-v3 on Uzbek validation/test splits.")
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--data-dir", default=str(Path.home() / "whisper-uz-ft/data"))
    parser.add_argument("--output", default=str(Path.home() / "whisper-uz-ft/outputs/baseline_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model, language="uz", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    out = {"model": args.model, "language": "uz", "task": "transcribe", "device": device}
    for split in args.splits:
        ds = load_csv(Path(args.data_dir) / f"{split}.csv")
        out[split] = evaluate_split(model, processor, ds, args.batch_size, args.beam_size, device, dtype)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
