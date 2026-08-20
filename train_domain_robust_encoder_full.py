#!/usr/bin/env python3
"""Train the E1 all-modality encoder on the complete Large cohort."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from domain_robust_encoder_model import DomainRobustPsgEncoder, MODALITY_CHANNELS


VERSION = "domain_robust_encoder_full_v20_e1da"
MODALITIES = tuple(MODALITY_CHANNELS)
SAMPLING_RATES = {"eeg": 128, "eog": 128, "ecg": 128, "resp": 32, "spo2": 1, "emg": 128}
STAGE_CODES = (1, 2, 3, 4, 5)
EVENT_NAMES = ("arousal", "respiratory", "limb")
EPOCHS_PER_STORAGE_CHUNK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", "--pilot-manifest", dest="manifest", type=Path, required=True)
    parser.add_argument("--annotation-cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("e0", "e1", "e11", "e12"), required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--windows-per-record", type=int, default=128)
    parser.add_argument("--natural-windows", type=int, default=64)
    parser.add_argument("--eval-windows-per-record", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--records-per-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
    parser.add_argument("--covariance-loss-weight", type=float, default=0.05)
    parser.add_argument("--variance-loss-weight", type=float, default=1.25)
    parser.add_argument("--site-adversary", action="store_true")
    parser.add_argument("--domain-reversal-max", type=float, default=0.02)
    parser.add_argument("--domain-warmup-epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--smoke-records", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def epoch_channel_valid(handle: h5py.File, modality: str, n_epochs: int) -> np.ndarray:
    channels = MODALITY_CHANNELS[modality]
    present = np.asarray(
        handle.get(f"quality/channel_present/{modality}", np.ones(channels, dtype=bool)),
        dtype=bool,
    )
    hard = np.asarray(handle[f"quality/channel_hard_valid_5s/{modality}"], dtype=bool)
    expected = n_epochs * 6
    if hard.shape != (expected, channels):
        raise RuntimeError(f"Invalid {modality} quality shape {hard.shape}")
    return present[None, :] & (hard.reshape(n_epochs, 6, channels).mean(axis=1) >= 4.0 / 6.0)


def safe_annotation(
    handle: h5py.File, path: str, n_epochs: int, dtype: Any, default: Any
) -> np.ndarray:
    if path not in handle:
        return np.full(n_epochs, default, dtype=dtype)
    values = np.asarray(handle[path], dtype=dtype)
    if values.shape != (n_epochs,):
        raise RuntimeError(f"Annotation shape mismatch at {path}: {values.shape}")
    return values


def scan_record(row: Mapping[str, Any], annotation_dir: Path) -> dict[str, Any]:
    record_id = str(row["record_id"])
    canonical_path = Path(str(row["canonical_path"]))
    annotation_path = annotation_dir / "records" / f"{record_id}.h5"
    with h5py.File(canonical_path, "r") as handle:
        n_epochs = int(handle.attrs["complete_epoch_count"])
        channel_valid = {
            modality: epoch_channel_valid(handle, modality, n_epochs)
            for modality in MODALITIES
        }
    modality_valid = np.column_stack(
        [np.any(channel_valid[modality], axis=1) for modality in MODALITIES]
    )
    neural_valid = modality_valid[:, 0] | modality_valid[:, 1]
    eligible = neural_valid & (modality_valid.sum(axis=1) >= 2)

    with h5py.File(annotation_path, "r") as handle:
        human_stage = safe_annotation(
            handle, "annotations/human/stage", n_epochs, np.int8, -1
        )
        human_stage_valid = safe_annotation(
            handle, "annotations/human/stage_valid", n_epochs, bool, False
        )
        caisr_stage = safe_annotation(
            handle, "annotations/caisr/stage", n_epochs, np.int8, -1
        )
        caisr_stage_valid = safe_annotation(
            handle, "annotations/caisr/stage_valid", n_epochs, bool, False
        )
        probabilities = []
        for name in ("wake", "n1", "n2", "n3", "rem"):
            probabilities.append(
                safe_annotation(
                    handle, f"annotations/caisr/prob_{name}", n_epochs, np.float32, np.nan
                )
            )
        caisr_probabilities = np.column_stack(probabilities).astype(np.float32)
        probability_valid = safe_annotation(
            handle,
            "annotations/caisr/stage_probability_valid",
            n_epochs,
            bool,
            False,
        )
        human_events = []
        human_event_valid = []
        caisr_events = []
        caisr_event_valid = []
        for name in EVENT_NAMES:
            human_events.append(
                safe_annotation(
                    handle,
                    f"annotations/human/{name}_positive_fraction",
                    n_epochs,
                    np.float32,
                    np.nan,
                )
            )
            human_event_valid.append(
                safe_annotation(
                    handle,
                    f"annotations/human/{name}_valid_ratio",
                    n_epochs,
                    np.float32,
                    0.0,
                )
                >= 0.5
            )
            caisr_events.append(
                safe_annotation(
                    handle,
                    f"annotations/caisr/{name}_positive_fraction",
                    n_epochs,
                    np.float32,
                    np.nan,
                )
            )
            caisr_event_valid.append(
                safe_annotation(
                    handle,
                    f"annotations/caisr/{name}_valid_ratio",
                    n_epochs,
                    np.float32,
                    0.0,
                )
                >= 0.5
            )

    combined_stage = np.where(human_stage_valid, human_stage, caisr_stage)
    combined_stage_valid = human_stage_valid | caisr_stage_valid
    human_events_array = np.column_stack(human_events).astype(np.float32)
    human_event_valid_array = np.column_stack(human_event_valid).astype(bool)
    caisr_events_array = np.column_stack(caisr_events).astype(np.float32)
    caisr_event_valid_array = np.column_stack(caisr_event_valid).astype(bool)
    event_positive = np.any(
        (np.nan_to_num(human_events_array) >= 0.10) & human_event_valid_array
        | (np.nan_to_num(caisr_events_array) >= 0.10) & caisr_event_valid_array,
        axis=1,
    )
    return {
        "record_id": record_id,
        "canonical_path": str(canonical_path),
        "n_epochs": n_epochs,
        "eligible": np.flatnonzero(eligible).astype(np.int32),
        "combined_stage": combined_stage.astype(np.int8),
        "combined_stage_valid": combined_stage_valid.astype(bool),
        "human_stage": human_stage.astype(np.int8),
        "human_stage_valid": human_stage_valid.astype(bool),
        "caisr_stage": caisr_stage.astype(np.int8),
        "caisr_stage_valid": (caisr_stage_valid & probability_valid).astype(bool),
        "caisr_probabilities": caisr_probabilities,
        "human_events": human_events_array,
        "human_event_valid": human_event_valid_array,
        "caisr_events": caisr_events_array,
        "caisr_event_valid": caisr_event_valid_array,
        "event_positive": event_positive.astype(bool),
        "modality_valid": modality_valid.astype(bool),
    }


def load_or_build_index(
    manifest: pd.DataFrame, annotation_dir: Path, path: Path
) -> dict[str, Any]:
    expected = set(manifest["record_id"].astype(str))
    if path.is_file():
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") == VERSION and set(payload.get("records", {})) == expected:
            print(f"Loaded candidate index: {path}", flush=True)
            return payload
    records: dict[str, Any] = {}
    started = time.time()
    for position, row in enumerate(manifest.to_dict(orient="records"), start=1):
        records[str(row["record_id"])] = scan_record(row, annotation_dir)
        if position == 1 or position % 100 == 0 or position == len(manifest):
            print(f"Indexed {position}/{len(manifest)} records", flush=True)
    payload = {"version": VERSION, "records": records, "elapsed_sec": time.time() - started}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def choose_unique(
    pool: Sequence[int] | np.ndarray,
    count: int,
    rng: np.random.Generator,
    used: set[int],
) -> list[int]:
    available = np.asarray([int(value) for value in pool if int(value) not in used], dtype=np.int32)
    if len(available) == 0 or count <= 0:
        return []
    chosen = available if len(available) <= count else rng.choice(available, count, replace=False)
    output = [int(value) for value in chosen]
    used.update(output)
    return output


def sample_record_windows(
    record: Mapping[str, Any], total: int, natural: int, seed: int
) -> np.ndarray:
    eligible = np.asarray(record["eligible"], dtype=np.int32)
    if len(eligible) == 0:
        return eligible
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    chosen = choose_unique(eligible, min(natural, total), rng, used)
    remaining = total - len(chosen)
    if remaining > 0:
        stage = np.asarray(record["combined_stage"])
        stage_valid = np.asarray(record["combined_stage_valid"])
        for code in STAGE_CODES:
            pool = eligible[stage_valid[eligible] & (stage[eligible] == code)]
            chosen.extend(choose_unique(pool, 6, rng, used))
    if len(chosen) < total:
        event_positive = np.asarray(record["event_positive"])
        chosen.extend(choose_unique(eligible[event_positive[eligible]], 18, rng, used))
    if len(chosen) < total:
        third = max(int(record["n_epochs"]) // 3, 1)
        for low, high in ((0, third), (third, 2 * third), (2 * third, int(record["n_epochs"]))):
            pool = eligible[(eligible >= low) & (eligible < high)]
            chosen.extend(choose_unique(pool, 5, rng, used))
    if len(chosen) < total:
        chosen.extend(choose_unique(eligible, total - len(chosen), rng, used))
    if len(chosen) < total:
        extra = rng.choice(eligible, size=total - len(chosen), replace=True).tolist()
        chosen.extend(int(value) for value in extra)
    rng.shuffle(chosen)
    return np.asarray(chosen[:total], dtype=np.int32)


def build_training_rows(
    records: Mapping[str, Mapping[str, Any]],
    record_ids: Sequence[str],
    total: int,
    natural: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[tuple[str, int]] = []
    for record_id in record_ids:
        indices = sample_record_windows(
            records[record_id], total, natural, stable_seed(seed, record_id)
        )
        rows.extend((record_id, int(index)) for index in indices)
    return pd.DataFrame(rows, columns=["record_id", "window_index"])


def build_eval_rows(
    records: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str], count: int
) -> pd.DataFrame:
    rows: list[tuple[str, int]] = []
    for record_id in sorted(record_ids):
        eligible = np.asarray(records[record_id]["eligible"], dtype=np.int32)
        if len(eligible) > count:
            positions = np.rint(np.linspace(0, len(eligible) - 1, count)).astype(int)
            eligible = eligible[positions]
        rows.extend((record_id, int(index)) for index in eligible)
    return pd.DataFrame(rows, columns=["record_id", "window_index"])


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def read_modality_windows(
    handle: h5py.File, modality: str, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rate = SAMPLING_RATES[modality]
    samples_per_epoch = rate * 30
    channels = MODALITY_CHANNELS[modality]
    indices = np.asarray(indices, dtype=np.int64)
    output = np.zeros((len(indices), channels, samples_per_epoch), dtype=np.float32)
    dataset = handle[f"signals/{modality}"]

    # Official-server preprocessing stores normalized, legacy-compatible int16
    # 30-second epochs. Earlier inference caches used float16, while the original
    # research archive stored the same quantized signal continuously. Support all
    # layouts without changing the training objective or model.
    if dataset.ndim == 3:
        if dataset.shape[1:] != (channels, samples_per_epoch):
            raise RuntimeError(
                f"Invalid online {modality} signal shape {dataset.shape}"
            )
        unique, inverse = np.unique(indices, return_inverse=True)
        output = np.asarray(dataset[unique], dtype=np.float32)[inverse]
        if np.issubdtype(dataset.dtype, np.integer):
            output /= 256.0
        n_epochs = int(handle.attrs["complete_epoch_count"])
        valid = epoch_channel_valid(handle, modality, n_epochs)[indices].astype(np.float32)
        output = np.nan_to_num(output, nan=0.0, posinf=12.0, neginf=-12.0)
        output = np.clip(output, -12.0, 12.0) * valid[:, :, None]
        return output, valid

    if dataset.ndim != 2:
        raise RuntimeError(f"Invalid legacy {modality} signal shape {dataset.shape}")
    storage_modes = decode_strings(np.asarray(handle[f"quantization/{modality}/storage_mode"]))
    channel_names = decode_strings(np.asarray(handle[f"metadata/{modality}/channel_names"]))
    centers = np.asarray(handle[f"quantization/{modality}/center"], dtype=np.float64)
    scales = np.asarray(handle[f"quantization/{modality}/robust_scale"], dtype=np.float64)
    for chunk_id in np.unique(indices // EPOCHS_PER_STORAGE_CHUNK):
        positions = np.flatnonzero(indices // EPOCHS_PER_STORAGE_CHUNK == chunk_id)
        sample_start = int(chunk_id) * EPOCHS_PER_STORAGE_CHUNK * samples_per_epoch
        sample_end = min(
            sample_start + EPOCHS_PER_STORAGE_CHUNK * samples_per_epoch,
            dataset.shape[1],
        )
        chunk = np.asarray(dataset[:, sample_start:sample_end], dtype=np.float32) / 256.0
        for channel_index, mode in enumerate(storage_modes):
            if mode != "float64_fallback":
                continue
            full_path = f"overflow/{modality}/{channel_names[channel_index]}/full_values"
            full = np.asarray(handle[full_path][sample_start:sample_end], dtype=np.float32)
            chunk[channel_index] = (
                full - centers[channel_index]
            ) / max(scales[channel_index], 1e-12)
        for position in positions:
            local_epoch = int(indices[position] % EPOCHS_PER_STORAGE_CHUNK)
            start = local_epoch * samples_per_epoch
            output[position] = chunk[:, start : start + samples_per_epoch]
    n_epochs = int(handle.attrs["complete_epoch_count"])
    valid = epoch_channel_valid(handle, modality, n_epochs)[indices].astype(np.float32)
    output = np.nan_to_num(output, nan=0.0, posinf=12.0, neginf=-12.0)
    output = np.clip(output, -12.0, 12.0) * valid[:, :, None]
    return output, valid


def load_signal_batch(path: str, indices: np.ndarray) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as handle:
        for modality in MODALITIES:
            signal, mask = read_modality_windows(handle, modality, indices)
            output[modality] = torch.from_numpy(signal)
            output[f"{modality}_mask"] = torch.from_numpy(mask)
    return output


def attach_targets(
    batch: dict[str, torch.Tensor], record: Mapping[str, Any], indices: np.ndarray
) -> None:
    batch["human_stage"] = torch.from_numpy(record["human_stage"][indices].astype(np.int64) - 1)
    batch["human_stage_valid"] = torch.from_numpy(record["human_stage_valid"][indices])
    batch["caisr_stage"] = torch.from_numpy(record["caisr_stage"][indices].astype(np.int64) - 1)
    batch["caisr_stage_valid"] = torch.from_numpy(record["caisr_stage_valid"][indices])
    batch["caisr_probabilities"] = torch.from_numpy(record["caisr_probabilities"][indices])
    batch["human_events"] = torch.from_numpy(record["human_events"][indices])
    batch["human_event_valid"] = torch.from_numpy(record["human_event_valid"][indices])
    batch["caisr_events"] = torch.from_numpy(record["caisr_events"][indices])
    batch["caisr_event_valid"] = torch.from_numpy(record["caisr_event_valid"][indices])
    batch["combined_stage"] = torch.from_numpy(record["combined_stage"][indices].astype(np.int64) - 1)
    batch["combined_stage_valid"] = torch.from_numpy(record["combined_stage_valid"][indices])
    if "site_index" in record:
        batch["site_index"] = torch.full(
            (len(indices),), int(record["site_index"]), dtype=torch.long
        )


class RecordBatchDataset(IterableDataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        records: Mapping[str, Mapping[str, Any]],
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.groups = [
            (str(record_id), np.sort(group["window_index"].to_numpy(dtype=np.int64)))
            for record_id, group in rows.groupby("record_id", sort=True)
        ]

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        worker_count = 1 if worker is None else int(worker.num_workers)
        order = np.arange(len(self.groups))
        rng = np.random.default_rng(stable_seed(self.seed, worker_id))
        if self.shuffle:
            rng.shuffle(order)
        for group_position in order[worker_id::worker_count]:
            record_id, indices = self.groups[int(group_position)]
            record = self.records[record_id]
            loaded = load_signal_batch(record["canonical_path"], indices)
            attach_targets(loaded, record, indices)
            positions = np.arange(len(indices))
            if self.shuffle:
                rng.shuffle(positions)
            for start in range(0, len(positions), self.batch_size):
                selected = positions[start : start + self.batch_size]
                batch = {key: value[selected] for key, value in loaded.items()}
                batch["record_id"] = record_id
                batch["window_index"] = torch.from_numpy(indices[selected])
                yield batch


class MixedRecordBatchDataset(IterableDataset):
    """Keep record-local HDF5 reads while forming representation batches across nights."""

    def __init__(
        self,
        rows: pd.DataFrame,
        records: Mapping[str, Mapping[str, Any]],
        batch_size: int,
        records_per_batch: int,
        seed: int,
    ) -> None:
        super().__init__()
        if batch_size % records_per_batch != 0:
            raise ValueError("batch-size must be divisible by records-per-batch")
        self.records = records
        self.batch_size = batch_size
        self.records_per_batch = records_per_batch
        self.seed = seed
        self.groups = [
            (str(record_id), np.sort(group["window_index"].to_numpy(dtype=np.int64)))
            for record_id, group in rows.groupby("record_id", sort=True)
        ]

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        worker_count = 1 if worker is None else int(worker.num_workers)
        rng = np.random.default_rng(stable_seed(self.seed, worker_id, "mixed"))
        order = np.arange(len(self.groups))
        rng.shuffle(order)
        blocks = [
            order[start : start + self.records_per_batch]
            for start in range(0, len(order), self.records_per_batch)
            if len(order[start : start + self.records_per_batch]) == self.records_per_batch
        ]
        per_record = self.batch_size // self.records_per_batch
        for block in blocks[worker_id::worker_count]:
            loaded_records: list[tuple[str, np.ndarray, dict[str, torch.Tensor], np.ndarray]] = []
            for group_position in block:
                record_id, indices = self.groups[int(group_position)]
                record = self.records[record_id]
                loaded = load_signal_batch(record["canonical_path"], indices)
                attach_targets(loaded, record, indices)
                positions = np.arange(len(indices))
                rng.shuffle(positions)
                loaded_records.append((record_id, indices, loaded, positions))
            steps = min(
                len(positions) // per_record
                for _, _, _, positions in loaded_records
            )
            for step in range(steps):
                pieces: list[dict[str, torch.Tensor]] = []
                record_ids: list[str] = []
                window_indices: list[torch.Tensor] = []
                for record_id, indices, loaded, positions in loaded_records:
                    selected = positions[step * per_record : (step + 1) * per_record]
                    pieces.append({key: value[selected] for key, value in loaded.items()})
                    record_ids.extend([record_id] * len(selected))
                    window_indices.append(torch.from_numpy(indices[selected]))
                batch = {
                    key: torch.cat([piece[key] for piece in pieces], dim=0)
                    for key in pieces[0]
                }
                batch["record_id"] = record_ids
                batch["window_index"] = torch.cat(window_indices)
                yield batch


def make_loader(
    rows: pd.DataFrame,
    records: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    records_per_batch = int(getattr(args, "records_per_batch", 1))
    if shuffle and records_per_batch > 1:
        dataset: IterableDataset = MixedRecordBatchDataset(
            rows,
            records,
            args.batch_size,
            records_per_batch,
            seed,
        )
    else:
        dataset = RecordBatchDataset(rows, records, args.batch_size, seed, shuffle)
    kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def clone_signal_batch(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].clone()
        for modality in MODALITIES
        for key in (modality, f"{modality}_mask")
    }


def apply_style(signal: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    batch_size, channels, _ = signal.shape
    gain = torch.exp(torch.empty(batch_size, channels, 1, device=signal.device).uniform_(-0.30, 0.30))
    offset = torch.empty(batch_size, channels, 1, device=signal.device).uniform_(-0.08, 0.08)
    kernel = 5 if sampling_rate >= 32 else 3
    smooth = F.avg_pool1d(signal, kernel_size=kernel, stride=1, padding=kernel // 2)
    low_mix = torch.empty(batch_size, channels, 1, device=signal.device).uniform_(0.0, 0.25)
    high_mix = torch.empty(batch_size, channels, 1, device=signal.device).uniform_(0.0, 0.12)
    styled = signal + low_mix * (smooth - signal) + high_mix * (signal - smooth)
    noise_scale = torch.empty(batch_size, channels, 1, device=signal.device).uniform_(0.005, 0.050)
    styled = gain * styled + offset + noise_scale * torch.randn_like(styled)
    step = torch.empty(batch_size, channels, 1, device=signal.device).uniform_(0.002, 0.020)
    return torch.clamp(torch.round(styled / step) * step, -12.0, 12.0)


def augment_view(
    batch: Mapping[str, Any],
    device_style: bool,
    temporal_mask: bool,
    channel_dropout: float,
    modality_dropout: float,
    multi_block_tokens: bool = False,
) -> dict[str, torch.Tensor]:
    output = clone_signal_batch(batch)
    for modality in MODALITIES:
        signal = output[modality]
        mask = output[f"{modality}_mask"]
        if device_style:
            signal = apply_style(signal, SAMPLING_RATES[modality])
        token_mask = torch.zeros(
            signal.shape[0], 8, dtype=torch.bool, device=signal.device
        )
        if temporal_mask and multi_block_tokens and signal.shape[-1] >= 8:
            for item in range(signal.shape[0]):
                starts = torch.randperm(8, device=signal.device)[:2]
                for start_token in starts.tolist():
                    width_tokens = int(torch.randint(1, 3, (1,), device=signal.device).item())
                    end_token = min(start_token + width_tokens, 8)
                    token_mask[item, start_token:end_token] = True
                for token in torch.nonzero(
                    token_mask[item], as_tuple=False
                ).flatten().tolist():
                    start = int(round(token * signal.shape[-1] / 8.0))
                    end = int(round((token + 1) * signal.shape[-1] / 8.0))
                    signal[item, :, start:end] = 0.0
        elif temporal_mask and signal.shape[-1] >= 8:
            for item in range(signal.shape[0]):
                width = max(1, int(signal.shape[-1] * float(torch.empty(1).uniform_(0.10, 0.25))))
                start = int(torch.randint(0, max(signal.shape[-1] - width + 1, 1), (1,)).item())
                signal[item, :, start : start + width] = 0.0
        if channel_dropout > 0:
            dropped = torch.rand_like(mask) < channel_dropout
            mask = mask * (~dropped)
        if modality_dropout > 0:
            dropped_modality = (torch.rand(signal.shape[0], 1, device=signal.device) < modality_dropout)
            mask = mask * (~dropped_modality)
        signal = signal * mask.unsqueeze(-1)
        output[modality] = signal
        output[f"{modality}_mask"] = mask
        output[f"{modality}_token_mask"] = token_mask
    return output


def masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    valid = valid.bool() & (target >= 0) & (target < logits.shape[1])
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def soft_stage_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    finite = torch.isfinite(target).all(dim=1)
    valid = valid.bool() & finite & (target.sum(dim=1) > 0.5)
    if not torch.any(valid):
        return logits.sum() * 0.0
    probabilities = target[valid] / target[valid].sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(probabilities * F.log_softmax(logits[valid], dim=1)).sum(dim=1).mean()


def masked_bce(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    valid = valid.bool() & torch.isfinite(target)
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid].clamp(0.0, 1.0))


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.detach().float()
    return 2.0 - 2.0 * F.cosine_similarity(
        prediction, target, dim=1, eps=1e-6
    ).mean()


def representation_loss(
    model: DomainRobustPsgEncoder,
    student: Mapping[str, Any],
    teacher: Mapping[str, Any],
    original_batch: Mapping[str, Any],
    student_batch: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    global_loss = cosine_loss(
        model.predict_projection(student["projection"]), teacher["projection"]
    )
    modality_losses = []
    for modality in MODALITIES:
        available = original_batch[f"{modality}_mask"].bool().any(dim=1)
        if torch.any(available):
            prediction = model.predict_modality(
                modality, student["modality_projections"][modality][available]
            )
            target = teacher["modality_projections"][modality][available]
            modality_losses.append(cosine_loss(prediction, target))
    modality_loss = torch.stack(modality_losses).mean() if modality_losses else global_loss * 0.0
    token_losses = []
    if args.variant in {"e11", "e12"}:
        for modality in MODALITIES:
            available = (
                original_batch[f"{modality}_mask"].bool().any(dim=1)
                & student_batch[f"{modality}_mask"].bool().any(dim=1)
            )
            selected = student_batch[f"{modality}_token_mask"].bool() & available[:, None]
            if torch.any(selected):
                prediction = model.predict_tokens(
                    modality, student["token_projections"][modality][selected]
                )
                target = teacher["token_projections"][modality][selected]
                token_losses.append(cosine_loss(prediction, target))
    token_loss = torch.stack(token_losses).mean() if token_losses else global_loss * 0.0
    embedding = student["embedding"].float()
    standard_deviation = torch.sqrt(embedding.var(dim=0, unbiased=False) + 1e-4)
    target_standard_deviation = 1.0 if args.variant == "e11" else 0.75
    variance_loss = F.relu(target_standard_deviation - standard_deviation).mean()
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(embedding.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / embedding.shape[1]
    total = 0.5 * global_loss + 0.5 * modality_loss
    if args.variant == "e11":
        total = (
            total
            + args.token_loss_weight * token_loss
            + args.covariance_loss_weight * covariance_loss
            + args.variance_loss_weight * variance_loss
        )
    else:
        total = total + 0.05 * variance_loss
        if args.variant == "e12":
            total = total + args.token_loss_weight * token_loss
    return total, {
        "latent_global": float(global_loss.detach()),
        "latent_modality": float(modality_loss.detach()),
        "variance": float(variance_loss.detach()),
        "latent_tokens": float(token_loss.detach()),
        "covariance": float(covariance_loss.detach()),
    }


def supervised_loss(
    output: Mapping[str, Any], batch: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, float]]:
    stage_logits = output["stage_logits"]
    event_logits = output["event_logits"]
    human_stage = masked_cross_entropy(
        stage_logits, batch["human_stage"], batch["human_stage_valid"]
    )
    caisr_stage = soft_stage_loss(
        stage_logits, batch["caisr_probabilities"], batch["caisr_stage_valid"]
    )
    human_events = masked_bce(
        event_logits, batch["human_events"], batch["human_event_valid"]
    )
    caisr_events = masked_bce(
        event_logits, batch["caisr_events"], batch["caisr_event_valid"]
    )
    total = human_stage + 0.5 * caisr_stage + 0.4 * human_events + 0.25 * caisr_events
    return total, {
        "human_stage": float(human_stage.detach()),
        "caisr_stage": float(caisr_stage.detach()),
        "human_events": float(human_events.detach()),
        "caisr_events": float(caisr_events.detach()),
    }


@torch.no_grad()
def update_teacher(
    model: nn.Module, teacher: nn.Module, decay: float
) -> None:
    for target, source in zip(teacher.parameters(), model.parameters()):
        target.data.mul_(decay).add_(source.data, alpha=1.0 - decay)
    for target, source in zip(teacher.buffers(), model.buffers()):
        target.copy_(source)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def reverse_gradient(values: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(values, float(strength))


def site_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    per_window = F.cross_entropy(logits, targets, reduction="none")
    return (per_window * class_weights[targets]).mean()


def domain_reversal_strength(
    epoch_index: int, total_epochs: int, warmup_epochs: int, maximum: float
) -> float:
    if maximum < 0.0:
        raise ValueError("Domain reversal maximum must be non-negative")
    if warmup_epochs < 0:
        raise ValueError("Domain warmup epochs must be non-negative")
    if epoch_index < warmup_epochs:
        return 0.0
    active_epochs = max(total_epochs - warmup_epochs, 1)
    progress = (epoch_index - warmup_epochs + 1) / active_epochs
    return float(maximum * min(max(progress, 0.0), 1.0))


def train_epoch(
    model: DomainRobustPsgEncoder,
    teacher: DomainRobustPsgEncoder,
    site_head: nn.Module | None,
    site_class_weights: torch.Tensor | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    epoch_index: int,
) -> dict[str, float]:
    model.train()
    teacher.eval()
    if site_head is not None:
        site_head.train()
    totals: dict[str, float] = {}
    batches = 0
    reversal_strength = domain_reversal_strength(
        epoch_index,
        args.epochs,
        args.domain_warmup_epochs,
        args.domain_reversal_max,
    )
    trainable_parameters = list(model.parameters())
    if site_head is not None:
        trainable_parameters.extend(site_head.parameters())
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        student_view = augment_view(
            batch,
            device_style=args.variant in {"e1", "e11", "e12"},
            temporal_mask=True,
            channel_dropout=0.10 if args.variant in {"e1", "e11", "e12"} else 0.05,
            modality_dropout=0.08 if args.variant in {"e1", "e11", "e12"} else 0.0,
            multi_block_tokens=args.variant in {"e11", "e12"},
        )
        teacher_view = augment_view(
            batch,
            device_style=args.variant in {"e1", "e11", "e12"},
            temporal_mask=False,
            channel_dropout=0.0,
            modality_dropout=0.0,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.amp):
            student_output = model(student_view)
            with torch.no_grad():
                teacher_output = teacher(teacher_view)
            supervised, supervised_parts = supervised_loss(student_output, batch)
            representation, representation_parts = representation_loss(
                model, student_output, teacher_output, batch, student_view, args
            )
            domain_loss = supervised.new_zeros(())
            domain_accuracy = math.nan
            if site_head is not None:
                if "site_index" not in batch or site_class_weights is None:
                    raise RuntimeError("Site-adversarial batch is missing site_index")
                site_logits = site_head(
                    reverse_gradient(student_output["embedding"], reversal_strength)
                )
                site_target = batch["site_index"].long()
                domain_loss = site_classification_loss(
                    site_logits, site_target, site_class_weights
                )
                domain_accuracy = float(
                    (site_logits.argmax(dim=1) == site_target)
                    .float()
                    .mean()
                    .detach()
                )
            loss = supervised + representation + domain_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss for record={raw_batch.get('record_id')} "
                f"windows={raw_batch.get('window_index')}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradient for record={raw_batch.get('record_id')} "
                f"windows={raw_batch.get('window_index')}"
            )
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < 1.0:
            raise FloatingPointError(f"AMP scale fell below 1.0: {scaler.get_scale()}")
        update_teacher(model, teacher, args.ema_decay)
        values = {"loss": float(loss.detach()), **supervised_parts, **representation_parts}
        if site_head is not None:
            values.update(
                {
                    "domain_loss": float(domain_loss.detach()),
                    "domain_accuracy": domain_accuracy,
                    "domain_reversal_strength": reversal_strength,
                }
            )
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    for name, module in (("student", model), ("teacher", teacher)):
        invalid = [
            parameter_name
            for parameter_name, parameter in module.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if invalid:
            raise FloatingPointError(f"Non-finite {name} parameters: {invalid[:10]}")
    if site_head is not None:
        invalid = [
            name
            for name, parameter in site_head.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if invalid:
            raise FloatingPointError(f"Non-finite site-head parameters: {invalid[:10]}")
    return {key: value / max(batches, 1) for key, value in totals.items()}


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    if valid.sum() < 10 or np.unique(target[valid] >= 0.1).size < 2:
        return math.nan
    return float(roc_auc_score(target[valid] >= 0.1, score[valid]))


@torch.no_grad()
def evaluate_tasks(
    model: DomainRobustPsgEncoder,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    stage_logits: list[np.ndarray] = []
    human_stage: list[np.ndarray] = []
    human_valid: list[np.ndarray] = []
    caisr_stage: list[np.ndarray] = []
    caisr_valid: list[np.ndarray] = []
    event_scores: list[np.ndarray] = []
    human_events: list[np.ndarray] = []
    human_event_valid: list[np.ndarray] = []
    caisr_events: list[np.ndarray] = []
    caisr_event_valid: list[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(amp):
            output = model(batch)
        stage_logits.append(output["stage_logits"].float().cpu().numpy())
        event_scores.append(torch.sigmoid(output["event_logits"]).float().cpu().numpy())
        human_stage.append(batch["human_stage"].cpu().numpy())
        human_valid.append(batch["human_stage_valid"].cpu().numpy())
        caisr_stage.append(batch["caisr_stage"].cpu().numpy())
        caisr_valid.append(batch["caisr_stage_valid"].cpu().numpy())
        human_events.append(batch["human_events"].cpu().numpy())
        human_event_valid.append(batch["human_event_valid"].cpu().numpy())
        caisr_events.append(batch["caisr_events"].cpu().numpy())
        caisr_event_valid.append(batch["caisr_event_valid"].cpu().numpy())
    logits = np.concatenate(stage_logits)
    predicted = logits.argmax(axis=1)
    h_stage = np.concatenate(human_stage)
    h_valid = np.concatenate(human_valid).astype(bool)
    c_stage = np.concatenate(caisr_stage)
    c_valid = np.concatenate(caisr_valid).astype(bool)
    scores = np.concatenate(event_scores)
    h_events = np.concatenate(human_events)
    h_event_mask = np.concatenate(human_event_valid).astype(bool)
    c_events = np.concatenate(caisr_events)
    c_event_mask = np.concatenate(caisr_event_valid).astype(bool)
    result = {
        "human_stage_macro_recall": float(
            recall_score(h_stage[h_valid], predicted[h_valid], labels=range(5), average="macro", zero_division=0)
        ) if h_valid.any() else math.nan,
        "caisr_stage_macro_recall": float(
            recall_score(c_stage[c_valid], predicted[c_valid], labels=range(5), average="macro", zero_division=0)
        ) if c_valid.any() else math.nan,
        "human_stage_balanced_accuracy": float(
            balanced_accuracy_score(h_stage[h_valid], predicted[h_valid])
        ) if h_valid.any() else math.nan,
    }
    for position, name in enumerate(EVENT_NAMES):
        result[f"human_{name}_auroc"] = safe_auc(
            np.where(h_event_mask[:, position], h_events[:, position], np.nan),
            scores[:, position],
        )
        result[f"caisr_{name}_auroc"] = safe_auc(
            np.where(c_event_mask[:, position], c_events[:, position], np.nan),
            scores[:, position],
        )
    return result


@torch.no_grad()
def export_eval_embeddings(
    model: DomainRobustPsgEncoder,
    loader: DataLoader,
    records: Mapping[str, Mapping[str, Any]],
    manifest: pd.DataFrame,
    device: torch.device,
    amp: bool,
    output_path: Path,
    seed: int,
) -> None:
    model.eval()
    metadata = manifest.set_index("record_id").to_dict(orient="index")
    if "evaluation_include" in manifest:
        eval_manifest = manifest.loc[manifest["evaluation_include"].fillna(False).astype(bool)]
    elif "pilot_split" in manifest:
        eval_manifest = manifest.loc[manifest["pilot_split"].eq("eval")]
    else:
        eval_manifest = manifest
    configurations = eval_manifest["channel_config"].astype(str).drop_duplicates().tolist()
    common_bits = "".join(
        "1" if all(config[position] == "1" for config in configurations) else "0"
        for position in range(sum(MODALITY_CHANNELS.values()))
    )
    core_masks: dict[str, torch.Tensor] = {}
    start = 0
    for modality in MODALITIES:
        width = MODALITY_CHANNELS[modality]
        core_masks[modality] = torch.tensor(
            [bit == "1" for bit in common_bits[start : start + width]],
            dtype=torch.float32,
            device=device,
        )
        start += width
    payload: dict[str, list[Any]] = {
        "embedding": [], "dropout_embedding": [], "stage_logits": [],
        "dropout_stage_logits": [], "event_logits": [], "dropout_event_logits": [],
        "core_embedding": [], "core_stage_logits": [],
        "human_events": [], "human_event_valid": [],
        "caisr_events": [], "caisr_event_valid": [],
        "record_id": [], "window_index": [],
        "site": [], "age": [], "sex": [], "race": [], "channel_config": [],
        "stage": [], "stage_valid": [],
    }
    for batch_number, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        torch.manual_seed(stable_seed(seed, "dropout", batch_number))
        dropout_view = augment_view(
            batch,
            device_style=False,
            temporal_mask=False,
            channel_dropout=0.15,
            modality_dropout=0.15,
        )
        core_view = clone_signal_batch(batch)
        for modality in MODALITIES:
            fixed = core_masks[modality][None, :]
            core_view[f"{modality}_mask"] = core_view[f"{modality}_mask"] * fixed
            core_view[modality] = (
                core_view[modality] * core_view[f"{modality}_mask"].unsqueeze(-1)
            )
        with autocast_context(amp):
            clean = model(batch)
            dropped = model(dropout_view)
            core = model(core_view)
        count = clean["embedding"].shape[0]
        record_id = str(raw_batch["record_id"])
        row = metadata[record_id]
        record = records[record_id]
        indices = raw_batch["window_index"].numpy()
        stage = record["combined_stage"][indices].astype(np.int8) - 1
        stage_valid = record["combined_stage_valid"][indices]
        payload["embedding"].append(clean["embedding"].float().cpu().numpy())
        payload["dropout_embedding"].append(dropped["embedding"].float().cpu().numpy())
        payload["stage_logits"].append(clean["stage_logits"].float().cpu().numpy())
        payload["dropout_stage_logits"].append(dropped["stage_logits"].float().cpu().numpy())
        payload["event_logits"].append(clean["event_logits"].float().cpu().numpy())
        payload["dropout_event_logits"].append(dropped["event_logits"].float().cpu().numpy())
        payload["core_embedding"].append(core["embedding"].float().cpu().numpy())
        payload["core_stage_logits"].append(core["stage_logits"].float().cpu().numpy())
        payload["human_events"].append(record["human_events"][indices])
        payload["human_event_valid"].append(record["human_event_valid"][indices])
        payload["caisr_events"].append(record["caisr_events"][indices])
        payload["caisr_event_valid"].append(record["caisr_event_valid"][indices])
        payload["record_id"].extend([record_id] * count)
        payload["window_index"].extend(indices.tolist())
        payload["site"].extend([str(row["SiteID"])] * count)
        payload["age"].extend([float(row["Age"])] * count)
        payload["sex"].extend([str(row["Sex"])] * count)
        payload["race"].extend([str(row["Race"])] * count)
        payload["channel_config"].extend([str(row["channel_config"])] * count)
        payload["stage"].extend(stage.tolist())
        payload["stage_valid"].extend(stage_valid.tolist())
    concatenated_keys = {
        "embedding", "dropout_embedding", "stage_logits", "dropout_stage_logits",
        "event_logits", "dropout_event_logits", "core_embedding", "core_stage_logits",
        "human_events", "human_event_valid", "caisr_events", "caisr_event_valid",
    }
    half_keys = {
            "embedding", "dropout_embedding", "stage_logits", "dropout_stage_logits",
            "event_logits", "dropout_event_logits", "core_embedding", "core_stage_logits",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arrays[key] = np.concatenate(value) if key in concatenated_keys else np.asarray(value)
        if key in half_keys:
            arrays[key] = arrays[key].astype(np.float16)
    arrays["core_channel_bits"] = np.asarray(common_bits)
    np.savez_compressed(output_path, **arrays)


def split_mask(frame: pd.DataFrame, split: str) -> pd.Series:
    flag = "training_include" if split == "train" else "evaluation_include"
    if flag in frame:
        return frame[flag].fillna(False).astype(bool)
    if "pilot_split" not in frame:
        raise RuntimeError(f"Manifest is missing {flag} and pilot_split")
    return frame["pilot_split"].eq(split)


def subset_smoke(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0:
        return frame
    parts = []
    per_site = max(2, count // frame["SiteID"].nunique())
    for site, group in frame.groupby("SiteID", sort=True):
        train_pool = group[split_mask(group, "train")]
        eval_pool = group[split_mask(group, "eval")]
        train = train_pool.sample(
            n=min(per_site, len(train_pool)),
            random_state=stable_seed(seed, site, "smoke_train"),
        )
        evaluation = eval_pool.sample(
            n=min(2, len(eval_pool)),
            random_state=stable_seed(seed, site, "smoke_eval"),
        )
        parts.extend([train, evaluation])
    return pd.concat(parts, ignore_index=True).drop_duplicates("record_id")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise RuntimeError("--resume and --overwrite are mutually exclusive")
    if not math.isfinite(args.domain_reversal_max) or args.domain_reversal_max < 0.0:
        raise ValueError("--domain-reversal-max must be finite and non-negative")
    if args.domain_warmup_epochs < 0:
        raise ValueError("--domain-warmup-epochs must be non-negative")
    set_seed(args.seed)
    manifest = pd.read_parquet(args.manifest).copy()
    manifest["record_id"] = manifest["record_id"].astype(str)
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    if manifest["record_id"].duplicated().any():
        raise RuntimeError("Manifest contains duplicate record_id values")
    index = load_or_build_index(manifest, args.annotation_cache_dir, args.candidate_index)
    if args.index_only:
        print("Index complete", flush=True)
        return
    frame = subset_smoke(manifest, args.smoke_records, args.seed)
    train_ids = frame.loc[split_mask(frame, "train"), "record_id"].drop_duplicates().tolist()
    eval_ids = frame.loc[split_mask(frame, "eval"), "record_id"].drop_duplicates().tolist()
    if not train_ids or not eval_ids:
        raise RuntimeError("Training or evaluation selection is empty")
    output_dir = args.output_dir
    checkpoint_path = output_dir / "checkpoint_last.pt"
    output_has_files = output_dir.exists() and any(output_dir.iterdir())
    if args.resume and not checkpoint_path.is_file():
        raise RuntimeError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if output_has_files and not args.overwrite and not args.resume:
        raise RuntimeError(f"Output exists: {output_dir}; pass --resume or --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = index["records"]
    training_sites = (
        frame.loc[split_mask(frame, "train"), "SiteID"]
        .fillna("__MISSING__")
        .astype(str)
    )
    site_names = sorted(training_sites.unique().tolist())
    site_to_index = {site: index for index, site in enumerate(site_names)}
    record_sites = (
        frame.set_index("record_id")["SiteID"]
        .fillna("__MISSING__")
        .astype(str)
        .to_dict()
    )
    for record_id in set(train_ids + eval_ids):
        records[record_id]["site_index"] = site_to_index[record_sites[record_id]]
    site_adversary_enabled = bool(args.site_adversary and len(site_names) >= 2)
    if args.site_adversary and not site_adversary_enabled:
        print("Site adversary disabled because fewer than two sites are available", flush=True)
    eval_rows = build_eval_rows(records, eval_ids, args.eval_windows_per_record)
    eval_loader = make_loader(eval_rows, records, args, args.seed, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DomainRobustPsgEncoder().to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    site_head: nn.Module | None = None
    site_class_weights: torch.Tensor | None = None
    site_weight_map: dict[str, float] = {}
    if site_adversary_enabled:
        site_head = nn.Sequential(
            nn.Linear(model.embedding_dimension, 64),
            nn.GELU(),
            nn.Linear(64, len(site_names)),
        ).to(device)
        site_counts = training_sites.value_counts().reindex(site_names).astype(float)
        balanced_weights = len(training_sites) / (len(site_names) * site_counts)
        site_weight_map = {
            site: float(balanced_weights.loc[site]) for site in site_names
        }
        site_class_weights = torch.tensor(
            [site_weight_map[site] for site in site_names],
            dtype=torch.float32,
            device=device,
        )
    optimizer_parameters = list(model.parameters())
    if site_head is not None:
        optimizer_parameters.extend(site_head.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    history: list[dict[str, Any]] = []
    start_epoch = 0
    manifest_fingerprint = hashlib.sha256(
        "\n".join(sorted(manifest["record_id"])).encode()
    ).hexdigest()
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint.get("version") != VERSION:
            raise RuntimeError(f"Checkpoint version mismatch: {checkpoint.get('version')}")
        if checkpoint.get("variant") != args.variant:
            raise RuntimeError(f"Checkpoint variant mismatch: {checkpoint.get('variant')}")
        if checkpoint.get("manifest_fingerprint") != manifest_fingerprint:
            raise RuntimeError("Checkpoint manifest does not match the requested manifest")
        if bool(checkpoint.get("site_adversary_enabled", False)) != site_adversary_enabled:
            raise RuntimeError("Checkpoint site-adversary setting does not match")
        if checkpoint.get("site_to_index", {}) != site_to_index:
            raise RuntimeError("Checkpoint site mapping does not match the manifest")
        if checkpoint.get("site_class_weights", {}) != site_weight_map:
            raise RuntimeError("Checkpoint site-class weights do not match the manifest")
        model.load_state_dict(checkpoint["model_state"])
        teacher.load_state_dict(checkpoint["teacher_state"])
        if site_head is not None:
            site_state = checkpoint.get("site_head_state")
            if not isinstance(site_state, dict) or not site_state:
                raise RuntimeError("Checkpoint is missing the site-head state")
            site_head.load_state_dict(site_state)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())
            history = [entry for entry in history if int(entry["epoch"]) <= start_epoch]
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resumed from epoch {start_epoch}", flush=True)
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        rows = build_training_rows(
            records,
            train_ids,
            args.windows_per_record,
            args.natural_windows,
            stable_seed(args.seed, "epoch", epoch),
        )
        loader = make_loader(rows, records, args, stable_seed(args.seed, epoch), shuffle=True)
        train_metrics = train_epoch(
            model,
            teacher,
            site_head,
            site_class_weights,
            loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
        )
        eval_metrics = evaluate_tasks(model, eval_loader, device, args.amp)
        entry = {
            "epoch": epoch + 1,
            "train_windows": int(len(rows)),
            "train": train_metrics,
            "eval": eval_metrics,
            "elapsed_sec": time.time() - started,
        }
        history.append(entry)
        history_tmp = output_dir / "history.json.tmp"
        history_tmp.write_text(json.dumps(history, indent=2) + "\n")
        history_tmp.replace(output_dir / "history.json")
        checkpoint_tmp = output_dir / f"checkpoint_epoch_{epoch + 1:02d}.pt.tmp"
        checkpoint_epoch = output_dir / f"checkpoint_epoch_{epoch + 1:02d}.pt"
        torch.save(
            {
                "version": VERSION,
                "variant": args.variant,
                "epoch": epoch + 1,
                "manifest_fingerprint": manifest_fingerprint,
                "site_adversary_enabled": site_adversary_enabled,
                "site_to_index": site_to_index,
                "site_class_weights": site_weight_map,
                "site_head_state": site_head.state_dict() if site_head is not None else None,
                "model_state": model.state_dict(),
                "teacher_state": teacher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "args": vars(args),
            },
            checkpoint_tmp,
        )
        checkpoint_tmp.replace(checkpoint_epoch)
        checkpoint_last_tmp = output_dir / "checkpoint_last.pt.tmp"
        shutil.copy2(checkpoint_epoch, checkpoint_last_tmp)
        checkpoint_last_tmp.replace(checkpoint_path)
        print(json.dumps(entry, sort_keys=True), flush=True)

    final_metrics = evaluate_tasks(model, eval_loader, device, args.amp)
    (output_dir / "task_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n"
    )
    export_loader = make_loader(eval_rows, records, args, args.seed, shuffle=False)
    export_eval_embeddings(
        model,
        export_loader,
        records,
        frame,
        device,
        args.amp,
        output_dir / "eval_embeddings.npz",
        args.seed,
    )
    summary = {
        "status": "complete",
        "variant": args.variant,
        "train_records": len(train_ids),
        "eval_records": len(eval_ids),
        "monitor_is_training_subset": bool(set(eval_ids).issubset(set(train_ids))),
        "site_adversary": {
            "enabled": site_adversary_enabled,
            "site_count": len(site_names) if site_adversary_enabled else 0,
            "warmup_epochs": args.domain_warmup_epochs,
            "maximum_reversal_strength": args.domain_reversal_max,
            "class_weights": site_weight_map,
            "class_weight_basis": "inverse_training_record_count",
            "training_only": True,
        },
        "epochs": args.epochs,
        "elapsed_sec": time.time() - started,
        "task_metrics": final_metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
