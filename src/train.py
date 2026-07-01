from __future__ import annotations

import argparse
import json
import os
import inspect
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import soundfile as sf
import torch
import yaml
import datasets
from datasets import DatasetDict, Features, Value, load_dataset
from torch.utils.data import Sampler, WeightedRandomSampler
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from model import (
    detailed_trainable_parameter_report,
    load_whisper_for_partial_ft,
    trainable_parameter_report,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and torch.all(labels[:, 0] == self.processor.tokenizer.bos_token_id):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class OnTheFlySpeechDataset:
    """CSV-backed speech dataset that computes Whisper features at item load time.

    This avoids storing large `input_features` Arrow cache files on disk. The
    underlying Hugging Face dataset only contains small metadata columns.
    """

    def __init__(
        self,
        base_dataset,
        processor,
        max_label_length: int,
        use_weighted_sampling: bool = False,
    ):
        self.base_dataset = base_dataset
        self.processor = processor
        self.max_label_length = max_label_length
        self.use_weighted_sampling = use_weighted_sampling
        self.column_names = list(base_dataset.column_names)
        for name in ("input_features", "labels", "input_length"):
            if name not in self.column_names:
                self.column_names.append(name)
        if use_weighted_sampling and "_sampling_weight" not in self.column_names:
            self.column_names.append("_sampling_weight")
        self._input_lengths = [float(value) for value in base_dataset["duration"]]
        self._sampling_weights = (
            [float(value) for value in base_dataset["sampling_weight"]]
            if use_weighted_sampling and "sampling_weight" in base_dataset.column_names
            else None
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index):
        if isinstance(index, str):
            if index == "input_length":
                return self._input_lengths
            if index == "_sampling_weight" and self._sampling_weights is not None:
                return self._sampling_weights
            return self.base_dataset[index]
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(len(self)))]
        if isinstance(index, (list, tuple, np.ndarray)):
            return [self[int(idx)] for idx in index]

        row = self.base_dataset[int(index)]
        try:
            audio = load_audio_array(row["audio_path"])
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio_path={row.get('audio_path')!r}") from exc
        item = dict(row)
        item["input_features"] = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        item["labels"] = self.processor.tokenizer(row["text"]).input_ids[: self.max_label_length]
        item["input_length"] = float(row["duration"])
        if self.use_weighted_sampling and self._sampling_weights is not None:
            item["_sampling_weight"] = float(row["sampling_weight"])
        return item

    def select(self, indices):
        selected = self.base_dataset.select(indices)
        return OnTheFlySpeechDataset(
            selected,
            processor=self.processor,
            max_label_length=self.max_label_length,
            use_weighted_sampling=self.use_weighted_sampling,
        )


class JsonProgressCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, **logs}, ensure_ascii=False) + "\n")


class BestModelSnapshotCallback(TrainerCallback):
    """Persist a standalone best-model snapshot whenever validation WER improves."""

    def __init__(self, best_model_dir: Path, metrics_dir: Path, processor):
        self.best_model_dir = best_model_dir
        self.metrics_dir = metrics_dir
        self.processor = processor
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.metrics_dir / "validation_metrics_history.jsonl"
        self.best_metrics_path = self.metrics_dir / "best_metrics.json"
        self.best_wer: float | None = None

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, model=None, **kwargs):
        metrics = metrics or {}
        eval_wer = metrics.get("eval_wer")
        eval_cer = metrics.get("eval_cer")
        is_best = False
        if eval_wer is not None:
            eval_wer = float(eval_wer)
            is_best = self.best_wer is None or eval_wer < self.best_wer
            if is_best:
                self.best_wer = eval_wer

        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": int(state.global_step or 0),
            "epoch": float(state.epoch or 0.0),
            "val_loss": metrics.get("eval_loss"),
            "val_wer": eval_wer,
            "val_cer": float(eval_cer) if eval_cer is not None else None,
            "is_best": is_best,
        }
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if is_best and model is not None:
            tmp_dir = self.best_model_dir.with_name(self.best_model_dir.name + ".tmp")
            old_dir = self.best_model_dir.with_name(self.best_model_dir.name + ".old")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(tmp_dir))
            self.processor.save_pretrained(str(tmp_dir))
            if old_dir.exists():
                shutil.rmtree(old_dir)
            if self.best_model_dir.exists():
                self.best_model_dir.rename(old_dir)
            tmp_dir.rename(self.best_model_dir)
            if old_dir.exists():
                shutil.rmtree(old_dir)
            best_payload = {
                **row,
                "best_model_dir": str(self.best_model_dir),
                "metric_for_best_model": "wer",
                "greater_is_better": False,
            }
            self.best_metrics_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"BEST_MODEL_UPDATED: step={row['step']} "
                f"val_wer={row['val_wer']} val_cer={row['val_cer']} path={self.best_model_dir}",
                flush=True,
            )


class DurationBucketedSampler(Sampler[int]):
    """Shuffle by large buckets, then order each bucket by duration to reduce padding."""

    def __init__(self, lengths: list[float], seed: int, bucket_size: int):
        self.lengths = lengths
        self.seed = seed
        self.bucket_size = max(1, bucket_size)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        buckets = [indices[start : start + self.bucket_size] for start in range(0, len(indices), self.bucket_size)]
        buckets = [sorted(bucket, key=lambda idx: self.lengths[idx], reverse=True) for bucket in buckets]
        order = torch.randperm(len(buckets), generator=generator).tolist()
        for bucket_idx in order:
            yield from buckets[bucket_idx]


class OptimizedSeq2SeqTrainer(Seq2SeqTrainer):
    """Single-node training sampler extensions for curriculum and duration bucketing."""

    def __init__(self, *args, duration_bucketed_sampling: bool = False, duration_bucket_size_multiplier: int = 50, **kwargs):
        self.duration_bucketed_sampling = duration_bucketed_sampling
        self.duration_bucket_size_multiplier = duration_bucket_size_multiplier
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if train_dataset is None or "_sampling_weight" not in train_dataset.column_names:
            if (
                self.duration_bucketed_sampling
                and train_dataset is not None
                and "input_length" in train_dataset.column_names
            ):
                if self.args.world_size != 1:
                    raise RuntimeError("Duration-bucketed sampling currently supports one training process only.")
                batch_size = max(1, int(self.args.train_batch_size))
                bucket_size = batch_size * max(1, int(self.duration_bucket_size_multiplier))
                return DurationBucketedSampler(
                    lengths=[float(value) for value in train_dataset["input_length"]],
                    seed=int(self.args.data_seed),
                    bucket_size=bucket_size,
                )
            return super()._get_train_sampler(train_dataset)
        if self.args.world_size != 1:
            raise RuntimeError("Weighted curriculum sampling currently supports one training process only.")
        weights = torch.as_tensor(train_dataset["_sampling_weight"], dtype=torch.double)
        generator = torch.Generator()
        generator.manual_seed(int(self.args.data_seed))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    snapshot: dict[str, Any] = {
        "cuda_available": True,
        "torch_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "torch_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [item.strip() for item in result.stdout.strip().split(",")]
        if len(values) >= 5:
            snapshot.update(
                {
                    "gpu_util_percent": float(values[0]),
                    "vram_used_mib": float(values[1]),
                    "vram_total_mib": float(values[2]),
                    "power_watts": float(values[3]),
                    "temperature_c": float(values[4]),
                }
            )
    except Exception as exc:
        snapshot["nvidia_smi_error"] = repr(exc)
    return snapshot


def has_repeated_phrase(text: str) -> bool:
    tokens = text.lower().split()
    if len(tokens) < 8:
        return False
    for ngram_size in (1, 2, 3):
        seen = 0
        previous: tuple[str, ...] | None = None
        for idx in range(0, len(tokens) - ngram_size + 1):
            current = tuple(tokens[idx : idx + ngram_size])
            if current == previous:
                seen += 1
                if seen >= 4:
                    return True
            else:
                seen = 0
            previous = current
    return False


def prediction_quality_indicators(prediction: str, reference: str) -> dict[str, Any]:
    pred_len = max(1, len(prediction.strip()))
    ref_len = max(1, len(reference.strip()))
    length_ratio = pred_len / ref_len
    turkish_chars = set("ıİöÖüÜçÇşŞğĞ")
    kazakh_chars = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
    latin = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '.,?!:-")
    uzbek_cyrillic = set("абвгдежзийклмнопрстуфхцчшъьэюяёўқғҳўАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЪЬЭЮЯЁЎҚҒҲ")
    chars = set(prediction)
    language_confusion = bool(chars & turkish_chars or chars & kazakh_chars)
    unexpected_chars = sorted(ch for ch in chars if ch not in latin and ch not in uzbek_cyrillic and not ch.isspace())
    hallucination = length_ratio > 3.0 or has_repeated_phrase(prediction)
    return {
        "prediction_length": pred_len,
        "reference_length": ref_len,
        "length_ratio": length_ratio,
        "language_confusion": language_confusion,
        "unexpected_chars": unexpected_chars[:20],
        "hallucination": hallucination,
    }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_config = config.pop("base_config", None)
    if not base_config:
        return config
    base_path = Path(base_config).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = load_config(base_path.resolve())
    merged.update(config)
    return merged


def find_latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def step(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    return str(max(checkpoints, key=step))


def verify_checkpoint(path: Path) -> tuple[bool, list[str]]:
    required = ["trainer_state.json", "training_args.bin"]
    problems: list[str] = []
    if not path.exists():
        return False, [f"missing checkpoint directory: {path}"]
    for name in required:
        item = path / name
        if not item.exists() or item.stat().st_size == 0:
            problems.append(f"missing or empty {item}")
    has_model = any((path / name).exists() and (path / name).stat().st_size > 0 for name in ["model.safetensors", "pytorch_model.bin"])
    if not has_model:
        problems.append(f"missing model weights in {path}")
    trainer_state = path / "trainer_state.json"
    if trainer_state.exists() and trainer_state.stat().st_size > 0:
        try:
            json.loads(trainer_state.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"invalid trainer_state.json: {exc!r}")
    return not problems, problems


ENCODER_BLOCK_SPECS = {
    "encoder_0_7": ("a", 0, 7),
    "encoder_8_15": ("b", 8, 15),
    "encoder_16_23": ("c", 16, 23),
    "encoder_24_31": ("d", 24, 31),
}


def encoder_block_lrs_from_config(cfg: dict) -> dict[str, float]:
    return {
        "a": float(cfg.get("encoder_block_a_lr", 0.0) or 0.0),
        "b": float(cfg.get("encoder_block_b_lr", 0.0) or 0.0),
        "c": float(cfg.get("encoder_block_c_lr", 0.0) or 0.0),
        "d": float(cfg.get("encoder_block_d_lr", 0.0) or 0.0),
    }


def encoder_block_trainable_from_config(cfg: dict) -> dict[str, bool]:
    explicit = {}
    for block in ("a", "b", "c", "d"):
        key = f"encoder_block_{block}_trainable"
        if key in cfg:
            explicit[block] = bool(cfg[key])
    return explicit


def startup_lr_report(cfg: dict) -> dict[str, float]:
    if cfg.get("tuning_mode") == "blockwise_encoder_lr":
        block_lrs = encoder_block_lrs_from_config(cfg)
        return {
            "encoder_0_7": block_lrs["a"],
            "encoder_8_15": block_lrs["b"],
            "encoder_16_23": block_lrs["c"],
            "encoder_24_31": block_lrs["d"],
            "decoder": float(cfg.get("decoder_learning_rate", cfg["learning_rate"])),
        }
    encoder_lr = float(cfg.get("encoder_learning_rate", cfg["learning_rate"]))
    decoder_lr = float(cfg.get("decoder_learning_rate", cfg["learning_rate"]))
    return {
        "encoder_0_7": encoder_lr,
        "encoder_8_15": encoder_lr,
        "encoder_16_23": encoder_lr,
        "encoder_24_31": encoder_lr,
        "decoder": decoder_lr,
    }


def encoder_param_group(name: str, blockwise: bool) -> str:
    if not name.startswith("model.encoder."):
        return "other"
    if not blockwise:
        return "encoder"
    match = re.match(r"model\.encoder\.layers\.(\d+)\.", name)
    if not match:
        return "encoder_other"
    layer_idx = int(match.group(1))
    if 0 <= layer_idx <= 7:
        return "encoder_0_7"
    if 8 <= layer_idx <= 15:
        return "encoder_8_15"
    if 16 <= layer_idx <= 23:
        return "encoder_16_23"
    if 24 <= layer_idx <= 31:
        return "encoder_24_31"
    raise ValueError(f"Unexpected Whisper encoder layer index in parameter {name!r}")


def build_layerwise_optimizer(model: torch.nn.Module, cfg: dict, output_dir: Path) -> torch.optim.Optimizer:
    encoder_lr = float(cfg.get("encoder_learning_rate", cfg["learning_rate"]))
    decoder_lr = float(cfg.get("decoder_learning_rate", cfg["learning_rate"]))
    block_lrs = encoder_block_lrs_from_config(cfg)
    blockwise = cfg.get("tuning_mode") == "blockwise_encoder_lr"
    weight_decay = float(cfg["weight_decay"])
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")

    grouped: dict[tuple[str, float], dict[str, Any]] = {}
    group_counts: dict[str, dict[str, int | float]] = {}
    seen: set[int] = set()

    def group_name(name: str) -> str:
        if name.startswith("model.encoder."):
            return encoder_param_group(name, blockwise=blockwise)
        if name.startswith("model.decoder.") or name.startswith("proj_out."):
            return "decoder"
        return "other"

    def group_lr(name: str) -> float:
        if blockwise:
            if name == "encoder_0_7":
                return block_lrs["a"]
            if name == "encoder_8_15":
                return block_lrs["b"]
            if name == "encoder_16_23":
                return block_lrs["c"]
            if name == "encoder_24_31":
                return block_lrs["d"]
            if name == "encoder_other":
                return encoder_lr
        return encoder_lr if name == "encoder" else decoder_lr

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        base_group = group_name(name)
        decay = 0.0 if any(term in name for term in no_decay_terms) else weight_decay
        lr = group_lr(base_group)
        if lr <= 0.0:
            raise RuntimeError(f"Trainable parameter {name!r} was assigned non-positive LR={lr}")
        key = (base_group, decay)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": decay,
                "name": f"{base_group}_{'decay' if decay else 'no_decay'}",
            }
        grouped[key]["params"].append(param)

        stats = group_counts.setdefault(
            grouped[key]["name"],
            {"parameters": 0, "tensors": 0, "lr": lr, "weight_decay": decay},
        )
        stats["parameters"] = int(stats["parameters"]) + int(param.numel())
        stats["tensors"] = int(stats["tensors"]) + 1

    optimizer_groups = list(grouped.values())
    if not optimizer_groups:
        raise RuntimeError("No trainable parameters found for optimizer construction.")

    report = {
        "optimizer": "AdamW",
        "encoder_learning_rate": encoder_lr,
        "decoder_learning_rate": decoder_lr,
        "encoder_block_lrs": {
            "encoder_0_7": block_lrs["a"],
            "encoder_8_15": block_lrs["b"],
            "encoder_16_23": block_lrs["c"],
            "encoder_24_31": block_lrs["d"],
        },
        "groups": group_counts,
    }
    (output_dir / "optimizer_param_groups.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return torch.optim.AdamW(
        optimizer_groups,
        betas=(float(cfg.get("adam_beta1", 0.9)), float(cfg.get("adam_beta2", 0.999))),
        eps=float(cfg.get("adam_epsilon", 1e-8)),
    )


class SafetyCallback(TrainerCallback):
    def __init__(
        self,
        max_observed_grad_norm: float = 5000.0,
        nan_loss_patience: int = 1,
        unsafe_grad_norm_patience: int = 3,
    ):
        self.max_observed_grad_norm = max_observed_grad_norm
        self.nan_loss_patience = nan_loss_patience
        self.unsafe_grad_norm_patience = unsafe_grad_norm_patience
        self.nan_loss_count = 0
        self.unsafe_grad_norm_count = 0

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            self.nan_loss_count += 1
            print(f"SAFETY_STOP: non-finite loss at step {state.global_step}: {loss}", flush=True)
            if self.nan_loss_count >= self.nan_loss_patience:
                control.should_save = True
                control.should_training_stop = True
        elif loss is not None:
            self.nan_loss_count = 0

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            grad_norm_f = float(grad_norm)
            if not math.isfinite(grad_norm_f) or grad_norm_f > self.max_observed_grad_norm:
                self.unsafe_grad_norm_count += 1
                if self.unsafe_grad_norm_count >= self.unsafe_grad_norm_patience:
                    print(
                        f"SAFETY_STOP: unsafe grad_norm at step {state.global_step}: "
                        f"{grad_norm} ({self.unsafe_grad_norm_count} consecutive observations)",
                        flush=True,
                    )
                    control.should_save = True
                    control.should_training_stop = True
                else:
                    print(
                        f"SAFETY_WARNING: unsafe grad_norm at step {state.global_step}: "
                        f"{grad_norm} ({self.unsafe_grad_norm_count}/"
                        f"{self.unsafe_grad_norm_patience} consecutive observations)",
                        flush=True,
                    )
            else:
                self.unsafe_grad_norm_count = 0

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ok, problems = verify_checkpoint(checkpoint)
        if ok:
            print(f"CHECKPOINT_OK: {checkpoint}", flush=True)
        else:
            print(f"CHECKPOINT_WARNING: {checkpoint}: {'; '.join(problems)}", flush=True)


class ProductionStatusCallback(TrainerCallback):
    def __init__(
        self,
        report_dir: Path,
        report_steps: set[int],
        processor,
        eval_dataset,
        data_collator,
        sample_prediction_count: int = 3,
        stop_on_two_eval_wer_worsen: bool = True,
        hallucination_stop_threshold: float = 0.1,
        hallucination_increase_threshold: float = 0.05,
    ):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.report_steps = report_steps
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.sample_prediction_count = sample_prediction_count
        self.stop_on_two_eval_wer_worsen = stop_on_two_eval_wer_worsen
        self.hallucination_stop_threshold = hallucination_stop_threshold
        self.hallucination_increase_threshold = hallucination_increase_threshold
        self.start_time = time.time()
        self.last_log: dict[str, Any] = {}
        self.eval_wer_history: list[float] = []
        self.hallucination_history: list[float] = []

    def _eta(self, state: TrainerState) -> dict[str, Any]:
        elapsed = time.time() - self.start_time
        max_steps = int(state.max_steps or 0)
        step = int(state.global_step or 0)
        if max_steps <= 0 or step <= 0:
            return {"elapsed_seconds": elapsed, "eta_seconds": None, "progress_percent": None}
        rate = elapsed / step
        remaining = max(0, max_steps - step)
        return {
            "elapsed_seconds": elapsed,
            "eta_seconds": remaining * rate,
            "progress_percent": step / max_steps * 100.0,
        }

    def _sample_predictions(self, model) -> list[dict[str, Any]]:
        if self.sample_prediction_count <= 0:
            return []
        was_training = model.training
        model.eval()
        rows = []
        count = min(self.sample_prediction_count, len(self.eval_dataset))
        with torch.no_grad():
            for idx in range(count):
                sample = self.eval_dataset[idx]
                batch = self.data_collator([sample])
                labels = batch["labels"].clone()
                batch = {key: value.to(model.device) for key, value in batch.items() if key != "labels"}
                generated = model.generate(**batch, max_length=int(getattr(model.generation_config, "max_length", 225) or 225), num_beams=1)
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                prediction = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                reference = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)[0]
                rows.append(
                    {
                        "index": idx,
                        "prediction": prediction,
                        "reference": reference,
                        "indicators": prediction_quality_indicators(prediction, reference),
                    }
                )
        if was_training:
            model.train()
        return rows

    def _write_report(
        self,
        state: TrainerState,
        report_type: str,
        metrics: dict[str, Any] | None = None,
        samples: list[dict[str, Any]] | None = None,
        stop_reason: str | None = None,
    ) -> None:
        payload = {
            "type": report_type,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": int(state.global_step or 0),
            "max_steps": int(state.max_steps or 0),
            "last_train_log": self.last_log,
            "metrics": metrics or {},
            "gpu": gpu_snapshot(),
            "eta": self._eta(state),
            "samples": samples or [],
            "stop_reason": stop_reason,
        }
        json_path = self.report_dir / f"status_step_{state.global_step}_{report_type}.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path = self.report_dir / f"status_step_{state.global_step}_{report_type}.md"
        md_lines = [
            f"# Full FT Status: step {state.global_step}",
            "",
            f"- Type: `{report_type}`",
            f"- Timestamp UTC: `{payload['timestamp_utc']}`",
            f"- Progress: `{payload['eta'].get('progress_percent')}`",
            f"- ETA seconds: `{payload['eta'].get('eta_seconds')}`",
            f"- Train loss: `{self.last_log.get('loss')}`",
            f"- Eval loss: `{(metrics or {}).get('eval_loss')}`",
            f"- Eval WER: `{(metrics or {}).get('eval_wer')}`",
            f"- Eval CER: `{(metrics or {}).get('eval_cer')}`",
            f"- Eval hallucination rate: `{(metrics or {}).get('eval_hallucination_rate')}`",
            f"- Eval language confusion rate: `{(metrics or {}).get('eval_language_confusion_rate')}`",
            f"- GPU: `{payload['gpu']}`",
        ]
        if stop_reason:
            md_lines.append(f"- Stop reason: `{stop_reason}`")
        if samples:
            md_lines.extend(["", "## Sample Predictions"])
            for sample in samples:
                md_lines.extend(
                    [
                        f"### Sample {sample['index']}",
                        f"- Prediction: `{sample['prediction']}`",
                        f"- Reference: `{sample['reference']}`",
                        f"- Indicators: `{sample['indicators']}`",
                    ]
                )
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"STATUS_REPORT: {md_path}", flush=True)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            self.last_log = {**self.last_log, **logs}
        if int(state.global_step or 0) in self.report_steps:
            self._write_report(state, "milestone")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, model=None, **kwargs):
        metrics = metrics or {}
        stop_reason = None
        eval_wer = metrics.get("eval_wer")
        if eval_wer is not None:
            self.eval_wer_history.append(float(eval_wer))
            if (
                self.stop_on_two_eval_wer_worsen
                and len(self.eval_wer_history) >= 3
                and self.eval_wer_history[-1] > self.eval_wer_history[-2] > self.eval_wer_history[-3]
            ):
                stop_reason = (
                    "eval_wer worsened for two consecutive evaluations: "
                    f"{self.eval_wer_history[-3:]}"
                )
                control.should_save = True
                control.should_training_stop = True

        hallucination_rate = metrics.get("eval_hallucination_rate")
        if hallucination_rate is not None:
            hallucination_rate = float(hallucination_rate)
            previous = self.hallucination_history[-1] if self.hallucination_history else 0.0
            self.hallucination_history.append(hallucination_rate)
            if (
                hallucination_rate >= self.hallucination_stop_threshold
                and hallucination_rate - previous >= self.hallucination_increase_threshold
            ):
                stop_reason = (
                    "hallucination rate increased substantially: "
                    f"previous={previous}, current={hallucination_rate}"
                )
                control.should_save = True
                control.should_training_stop = True

        samples = self._sample_predictions(model) if model is not None else []
        self._write_report(state, "eval", metrics=metrics, samples=samples, stop_reason=stop_reason)


def load_audio_array(path: str, target_sr: int = 16000) -> np.ndarray:
    try:
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


def build_dataset(
    data_dir: Path,
    processor,
    max_label_length: int,
    split_prefix: str = "",
    num_proc: int | None = None,
    include_test: bool = True,
    use_weighted_sampling: bool = False,
    disable_dataset_cache: bool = True,
) -> dict[str, OnTheFlySpeechDataset]:
    if disable_dataset_cache:
        datasets.disable_caching()
        print("HF dataset caching disabled", flush=True)
    prefix = f"{split_prefix}_" if split_prefix else ""
    data_files: dict[str, str] = {
        "train": str(data_dir / f"{prefix}train.csv"),
        "validation": str(data_dir / f"{prefix}val.csv"),
    }
    if include_test:
        data_files["test"] = str(data_dir / f"{prefix}test.csv")
    feature_spec = {
        "audio_path": Value("string"),
        "text": Value("string"),
        "duration": Value("float64"),
        "speaker_id": Value("string"),
        "split": Value("string"),
        "source_metadata": Value("string"),
    }
    if use_weighted_sampling:
        feature_spec.update(
            {
                "dataset_id": Value("string"),
                "tier": Value("string"),
                "trust_weight": Value("float64"),
                "quality_score": Value("float64"),
                "sampling_weight": Value("float64"),
            }
        )
    features = Features(feature_spec)
    ds = load_dataset("csv", data_files=data_files, features=features, keep_in_memory=True)

    wrapped: dict[str, OnTheFlySpeechDataset] = {}
    for split, split_dataset in ds.items():
        wrapped[split] = OnTheFlySpeechDataset(
            split_dataset,
            processor=processor,
            max_label_length=max_label_length,
            use_weighted_sampling=use_weighted_sampling and split == "train",
        )
    if num_proc and num_proc > 1:
        print(
            "FEATURE_PREPROCESSING_NUM_WORKERS_IGNORED: on-the-fly feature extraction avoids persistent cache",
            flush=True,
        )
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper large-v3 for Uzbek-only ASR.")
    parser.add_argument("--config", default=str(Path.home() / "whisper-uz-ft/configs/train.yaml"))
    parser.add_argument("--resume", default=None, help="Checkpoint path, 'auto', or empty for config value.")
    parser.add_argument("--sanity-check", action="store_true", help="Initialize the full pipeline and run one forward pass without training.")
    parser.add_argument("--sanity-report", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser())
    data_dir = Path(cfg["data_dir"]).expanduser()
    output_dir = Path(cfg["output_dir"]).expanduser()
    logging_dir = Path(cfg["logging_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("disable_hf_dataset_cache", True)):
        datasets.disable_caching()
    print(
        "HF_CACHE_PATHS: "
        f"HF_HOME={os.environ.get('HF_HOME')} "
        f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')} "
        f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}",
        flush=True,
    )
    if bool(cfg.get("disable_hf_dataset_cache", True)):
        print("HF dataset caching disabled", flush=True)

    if not torch.cuda.is_available() and cfg.get("require_cuda", True):
        raise RuntimeError("CUDA is not available. Set require_cuda: false in configs/train.yaml to run on CPU.")

    bundle = load_whisper_for_partial_ft(
        cfg["model_name"],
        language=cfg.get("language", "uz"),
        task=cfg.get("task", "transcribe"),
        train_last_encoder_blocks=cfg.get("train_last_encoder_blocks", 8),
        tuning_mode=cfg.get("tuning_mode"),
        encoder_block_lrs=encoder_block_lrs_from_config(cfg),
        encoder_block_trainable=encoder_block_trainable_from_config(cfg),
        decoder_learning_rate=float(cfg.get("decoder_learning_rate", cfg["learning_rate"])),
        decoder_trainable=bool(cfg.get("decoder_trainable", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    model = bundle.model
    processor = bundle.processor
    if cfg.get("apply_spec_augment", False):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = float(cfg.get("mask_time_prob", 0.03))
        model.config.mask_time_length = int(cfg.get("mask_time_length", 10))
        model.config.mask_time_min_masks = int(cfg.get("mask_time_min_masks", 2))
        model.config.mask_feature_prob = float(cfg.get("mask_feature_prob", 0.03))
        model.config.mask_feature_length = int(cfg.get("mask_feature_length", 16))
        model.config.mask_feature_min_masks = int(cfg.get("mask_feature_min_masks", 1))
    report = trainable_parameter_report(model)
    (output_dir / "trainable_parameters.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    detailed_report = detailed_trainable_parameter_report(model)
    (output_dir / "trainable_parameter_groups.json").write_text(
        json.dumps(detailed_report, indent=2), encoding="utf-8"
    )
    print("TRAINABLE PARAM SUMMARY", flush=True)
    lr_report = startup_lr_report(cfg)
    for group in ("encoder_0_7", "encoder_8_15", "encoder_16_23", "encoder_24_31", "decoder"):
        values = detailed_report[group]
        print(
            f"- {group}: {values['state']} "
            f"lr={lr_report[group]:.3g} "
            f"({values['trainable_parameters']:,}/{values['total_parameters']:,})",
            flush=True,
        )
    print(
        f"- trainable params: {report['trainable_parameters']:,}\n"
        f"- frozen params: {report['total_parameters'] - report['trainable_parameters']:,}\n"
        f"- trainable %: {report['trainable_percent']:.4f}",
        flush=True,
    )

    dataset = build_dataset(
        data_dir,
        processor,
        int(cfg.get("max_label_length", 448)),
        split_prefix=str(cfg.get("split_prefix") or ""),
        num_proc=int(cfg.get("feature_preprocessing_num_workers", 1)),
        include_test=bool(cfg.get("load_test_split", True)),
        use_weighted_sampling=bool(cfg.get("use_weighted_sampling", False)),
        disable_dataset_cache=bool(cfg.get("disable_hf_dataset_cache", True)),
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        indicators = [prediction_quality_indicators(prediction, reference) for prediction, reference in zip(pred_str, label_str)]
        hallucination_rate = float(np.mean([item["hallucination"] for item in indicators])) if indicators else 0.0
        language_confusion_rate = float(np.mean([item["language_confusion"] for item in indicators])) if indicators else 0.0
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
            "hallucination_rate": hallucination_rate,
            "language_confusion_rate": language_confusion_rate,
        }

    training_arg_values = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(cfg["per_device_batch_size"]),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", cfg["per_device_batch_size"])),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "warmup_steps": int(cfg.get("warmup_steps", 0)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.0)),
        "num_train_epochs": float(cfg["epochs"]),
        "weight_decay": float(cfg["weight_decay"]),
        "max_grad_norm": float(cfg["max_grad_norm"]),
        "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "fp16": bool(cfg.get("fp16", False) and torch.cuda.is_available()),
        "bf16": bool(cfg.get("bf16", False) and torch.cuda.is_available()),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": int(cfg["eval_steps"]),
        "save_steps": int(cfg["save_steps"]),
        "logging_steps": int(cfg["logging_steps"]),
        "predict_with_generate": True,
        "generation_max_length": int(cfg.get("generation_max_length", 225)),
        "generation_num_beams": int(cfg.get("generation_num_beams", 5)),
        "load_best_model_at_end": True,
        "metric_for_best_model": str(cfg.get("metric_for_best_model", "wer")),
        "greater_is_better": bool(cfg.get("greater_is_better", False)),
        "lr_scheduler_type": cfg["scheduler"],
        "dataloader_num_workers": int(cfg["dataloader_num_workers"]),
        "logging_dir": str(logging_dir),
        "report_to": ["tensorboard"],
        "save_total_limit": int(cfg.get("save_total_limit", 3)),
        "remove_unused_columns": False,
        "max_steps": int(cfg["max_steps"]) if cfg.get("max_steps") else -1,
        "seed": int(cfg.get("seed", 1729)),
        "data_seed": int(cfg.get("data_seed", cfg.get("seed", 1729))),
    }
    training_sig = inspect.signature(Seq2SeqTrainingArguments)
    if "length_column_name" in training_sig.parameters:
        training_arg_values["length_column_name"] = str(cfg.get("length_column_name", "input_length"))
    if "dataloader_persistent_workers" in training_sig.parameters:
        training_arg_values["dataloader_persistent_workers"] = bool(cfg.get("dataloader_persistent_workers", False))
    if "dataloader_prefetch_factor" in training_sig.parameters and int(cfg["dataloader_num_workers"]) > 0:
        training_arg_values["dataloader_prefetch_factor"] = int(cfg.get("dataloader_prefetch_factor", 2))
    training_args = Seq2SeqTrainingArguments(**training_arg_values)

    trainer_kwargs = {
        "args": training_args,
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "optimizers": (build_layerwise_optimizer(model, cfg, output_dir), None),
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_patience", 5))),
            JsonProgressCallback(logging_dir / "training_metrics.jsonl"),
            BestModelSnapshotCallback(
                best_model_dir=Path(cfg.get("best_model_dir", output_dir / "best_model")).expanduser(),
                metrics_dir=Path(cfg.get("metrics_dir", output_dir / "metrics")).expanduser(),
                processor=processor,
            ),
            SafetyCallback(
                max_observed_grad_norm=float(cfg.get("max_observed_grad_norm", 5000)),
                nan_loss_patience=int(cfg.get("nan_loss_patience", 1)),
                unsafe_grad_norm_patience=int(cfg.get("unsafe_grad_norm_patience", 3)),
            ),
            ProductionStatusCallback(
                report_dir=Path(cfg.get("status_report_dir", logging_dir / "status_reports")).expanduser(),
                report_steps={int(step) for step in cfg.get("status_report_steps", [])},
                processor=processor,
                eval_dataset=dataset["validation"],
                data_collator=data_collator,
                sample_prediction_count=int(cfg.get("sample_prediction_count", 3)),
                stop_on_two_eval_wer_worsen=bool(cfg.get("stop_on_two_eval_wer_worsen", True)),
                hallucination_stop_threshold=float(cfg.get("hallucination_stop_threshold", 0.1)),
                hallucination_increase_threshold=float(cfg.get("hallucination_increase_threshold", 0.05)),
            ),
        ],
    }
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = OptimizedSeq2SeqTrainer(
        **trainer_kwargs,
        duration_bucketed_sampling=bool(cfg.get("duration_bucketed_sampling", False)),
        duration_bucket_size_multiplier=int(cfg.get("duration_bucket_size_multiplier", 50)),
    )

    if args.sanity_check:
        sample = dataset["train"].select(range(1))
        batch = data_collator([sample[0]])
        batch = {key: value.to(model.device) for key, value in batch.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
        sanity_report = {
            "status": "ok",
            "model_device": str(model.device),
            "train_rows": int(len(dataset["train"])),
            "validation_rows": int(len(dataset["validation"])),
            "test_rows": int(len(dataset["test"])) if "test" in dataset else 0,
            "forward_loss": float(outputs.loss.detach().cpu()) if outputs.loss is not None else None,
            "trainable_parameters": report,
            "cuda_available": torch.cuda.is_available(),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
        }
        if args.sanity_report:
            Path(args.sanity_report).expanduser().write_text(json.dumps(sanity_report, indent=2), encoding="utf-8")
        print(json.dumps(sanity_report, indent=2))
        return

    dry_report = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        first_log_len = len(trainer.state.log_history)
        resume_value = args.resume if args.resume is not None else cfg.get("resume_from_checkpoint")
        if resume_value == "":
            resume_value = None
        if resume_value == "auto":
            resume_value = find_latest_checkpoint(output_dir)
        if resume_value:
            ok, problems = verify_checkpoint(Path(resume_value))
            if not ok:
                raise RuntimeError(f"Refusing to resume from corrupt checkpoint {resume_value}: {'; '.join(problems)}")
            print(f"RESUME_FROM_CHECKPOINT: {resume_value}", flush=True)
        trainer.train(resume_from_checkpoint=resume_value)
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            print(
                json.dumps(
                    {
                        "error": "CUDA OOM",
                        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    },
                    indent=2,
                ),
                flush=True,
            )
        raise RuntimeError(
            "CUDA OOM. Reduce per_device_batch_size to 1 or increase gradient_accumulation_steps in configs/train.yaml."
        ) from exc
    except RuntimeError as exc:
        print(f"TRAINING_RUNTIME_ERROR: {exc!r}", flush=True)
        raise

    losses = [x["loss"] for x in trainer.state.log_history[first_log_len:] if "loss" in x]
    dry_report = {
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "global_step": trainer.state.global_step,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
        "train_runtime": trainer.state.log_history[-1].get("train_runtime") if trainer.state.log_history else None,
        "train_samples_per_second": trainer.state.log_history[-1].get("train_samples_per_second")
        if trainer.state.log_history
        else None,
    }
    if cfg.get("dry_run_report"):
        Path(cfg["dry_run_report"]).expanduser().write_text(json.dumps(dry_report, indent=2), encoding="utf-8")

    trainer.save_model(str(output_dir / "final_model"))
    processor.save_pretrained(str(output_dir / "final_model"))
    trainer.remove_callback(EarlyStoppingCallback)
    test_metrics = {}
    if bool(cfg.get("evaluate_test_after_training", True)):
        if "test" not in dataset:
            raise RuntimeError("evaluate_test_after_training=true requires load_test_split=true")
        test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    run_metrics = {
        "train_summary": dry_report,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "final_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
        "gpu": gpu_snapshot(),
    }
    (output_dir / "run_metrics.json").write_text(
        json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"dry_report": dry_report, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
