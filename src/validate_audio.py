from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from data_loader import AUDIO_EXTS, load_dataset_flexible


def inspect_audio(path: str) -> dict:
    result = {"path": path, "readable": False}
    try:
        import soundfile as sf

        info = sf.info(path)
        result.update(
            {
                "readable": True,
                "duration": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "backend": "soundfile",
            }
        )
        return result
    except Exception as exc:
        result["soundfile_error"] = repr(exc)

    try:
        import torchaudio

        metadata = torchaudio.info(path)
        result.update(
            {
                "readable": True,
                "duration": float(metadata.num_frames) / float(metadata.sample_rate) if metadata.sample_rate else 0.0,
                "sample_rate": int(metadata.sample_rate),
                "channels": int(metadata.num_channels),
                "backend": "torchaudio",
            }
        )
        return result
    except Exception as exc:
        result["torchaudio_error"] = repr(exc)

    try:
        import librosa

        duration = librosa.get_duration(path=path)
        y, sr = librosa.load(path, sr=None, mono=False, duration=0.25)
        channels = 1 if np.ndim(y) == 1 else int(np.asarray(y).shape[0])
        result.update(
            {
                "readable": True,
                "duration": float(duration),
                "sample_rate": int(sr),
                "channels": channels,
                "backend": "librosa",
            }
        )
        return result
    except Exception as exc:
        result["librosa_error"] = repr(exc)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate random audio files.")
    parser.add_argument("--dataset-dir", default=str(Path.home() / "datasets/usc"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--report", default=str(Path.home() / "whisper-uz-ft/logs/audio_validation_report.json"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_dir.exists():
        report = {"dataset_dir": str(dataset_dir), "error": "dataset directory missing"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    try:
        frame = load_dataset_flexible(dataset_dir)
        paths = list(dict.fromkeys(map(str, frame["audio_path"].tolist())))
    except Exception:
        paths = [str(p) for p in dataset_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]

    rng = random.Random(42)
    sample = rng.sample(paths, min(args.sample_size, len(paths))) if paths else []
    results = [inspect_audio(p) for p in sample]
    valid = [r for r in results if r.get("readable")]
    durations = [float(r["duration"]) for r in valid]
    report = {
        "dataset_dir": str(dataset_dir),
        "sample_size": len(sample),
        "valid_count": len(valid),
        "corrupted_count": len(results) - len(valid),
        "duration_distribution": {
            "min": min(durations) if durations else None,
            "p50": float(np.percentile(durations, 50)) if durations else None,
            "p95": float(np.percentile(durations, 95)) if durations else None,
            "max": max(durations) if durations else None,
        },
        "sample_rate_distribution": dict(Counter(str(r.get("sample_rate")) for r in valid)),
        "channel_distribution": dict(Counter(str(r.get("channels")) for r in valid)),
        "backend_distribution": dict(Counter(str(r.get("backend")) for r in valid)),
        "total_hours_in_sample": float(sum(durations) / 3600.0),
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
