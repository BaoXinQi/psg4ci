#!/usr/bin/env python3
"""Raw EDF to frozen E1 embeddings to full-night CI inference."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

import online_features
from domain_robust_encoder_model import DomainRobustPsgEncoder, MODALITY_CHANNELS
from sequence_ci_model import (
    EMBEDDING_DIMENSION,
    MAX_BLOCKS,
    WINDOWS_PER_BLOCK,
    LocalFullNightTransformer,
)


MODALITIES = tuple(MODALITY_CHANNELS)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_runtime(model_root: Path, threads: int | None = None) -> dict[str, Any]:
    thread_count = threads or max(1, min(8, os.cpu_count() or 1))
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    encoder_checkpoint = _torch_load(model_root / "e1_encoder.pt")
    encoder_state = encoder_checkpoint.get(
        "model_state", encoder_checkpoint.get("teacher_state", encoder_checkpoint)
    )
    encoder = DomainRobustPsgEncoder().cpu().eval()
    encoder.load_state_dict(encoder_state, strict=True)

    metadata = json.loads((model_root / "metadata.json").read_text(encoding="utf-8"))
    sequence_files = metadata.get("sequence_files", ["raw_sequence_final.pt"])
    if not sequence_files:
        raise RuntimeError("No sequence ensemble members were configured")
    sequences: list[LocalFullNightTransformer] = []
    sequence_paths: list[Path] = []
    for filename in sequence_files:
        sequence_path = model_root / str(filename)
        sequence_checkpoint = _torch_load(sequence_path)
        sequence = LocalFullNightTransformer(
            demographic_dimension=1,
            use_demographics=False,
            d_model=256,
            dropout=0.15,
        ).cpu().eval()
        sequence.load_state_dict(sequence_checkpoint["model_state"], strict=True)
        sequences.append(sequence)
        sequence_paths.append(sequence_path)
    return {
        "model_root": model_root,
        "encoder": encoder,
        "sequences": sequences,
        "sequence_paths": sequence_paths,
        "encoder_batch_size": 64,
    }


def _epoch_channel_valid(
    handle: h5py.File, modality: str, epoch_count: int
) -> np.ndarray:
    channels = MODALITY_CHANNELS[modality]
    present = np.asarray(
        handle.get(f"quality/channel_present/{modality}", np.ones(channels, dtype=bool)),
        dtype=bool,
    )
    hard = np.asarray(handle[f"quality/channel_hard_valid_5s/{modality}"], dtype=bool)
    if hard.shape != (epoch_count * 6, channels):
        raise RuntimeError(f"Invalid {modality} quality shape: {hard.shape}")
    return present[None, :] & (
        hard.reshape(epoch_count, 6, channels).mean(axis=1) >= 4.0 / 6.0
    )


def eligible_windows(cache_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    with h5py.File(cache_path, "r") as handle:
        epoch_count = int(handle.attrs["complete_epoch_count"])
        channel_valid = {
            modality: _epoch_channel_valid(handle, modality, epoch_count)
            for modality in MODALITIES
        }
    modality_valid = np.column_stack(
        [np.any(channel_valid[modality], axis=1) for modality in MODALITIES]
    )
    neural_valid = modality_valid[:, 0] | modality_valid[:, 1]
    eligible = np.flatnonzero(neural_valid & (modality_valid.sum(axis=1) >= 2))
    return eligible.astype(np.int64), channel_valid, epoch_count


def encode_cache(
    cache_path: Path,
    encoder: DomainRobustPsgEncoder,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    indices, channel_valid, epoch_count = eligible_windows(cache_path)
    if len(indices) == 0:
        return indices, np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), epoch_count

    encoded: list[np.ndarray] = []
    with h5py.File(cache_path, "r") as handle, torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch: dict[str, torch.Tensor] = {}
            for modality in MODALITIES:
                values = np.asarray(
                    handle[f"signals/{modality}"][batch_indices], dtype=np.float32
                )
                mask = channel_valid[modality][batch_indices].astype(np.float32)
                values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
                values = np.clip(values, -12.0, 12.0) * mask[:, :, None]
                batch[modality] = torch.from_numpy(values)
                batch[f"{modality}_mask"] = torch.from_numpy(mask)
            # CI inference only consumes the fused 192D representation. Avoid
            # executing the SSL projection and semantic heads used in pretraining.
            embedding = encoder.encode(batch)[0]
            if not torch.isfinite(embedding).all():
                raise FloatingPointError("Non-finite online E1 embedding")
            encoded.append(embedding.cpu().numpy().astype(np.float32))
    return indices, np.concatenate(encoded), epoch_count


def build_night_tensors(
    window_index: np.ndarray,
    embeddings: np.ndarray,
    epoch_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_count = min(max(math.ceil(max(epoch_count, 1) / WINDOWS_PER_BLOCK), 1), MAX_BLOCKS)
    blocks = np.zeros(
        (1, block_count, WINDOWS_PER_BLOCK, EMBEDDING_DIMENSION), dtype=np.float32
    )
    mask = np.zeros((1, block_count, WINDOWS_PER_BLOCK), dtype=bool)
    valid = np.asarray(window_index) < block_count * WINDOWS_PER_BLOCK
    if np.any(valid):
        selected = np.asarray(window_index)[valid]
        blocks[0, selected // WINDOWS_PER_BLOCK, selected % WINDOWS_PER_BLOCK] = embeddings[valid]
        mask[0, selected // WINDOWS_PER_BLOCK, selected % WINDOWS_PER_BLOCK] = True
    return torch.from_numpy(blocks), torch.from_numpy(mask)


def sequence_logit(
    sequence: LocalFullNightTransformer,
    window_index: np.ndarray,
    embeddings: np.ndarray,
    epoch_count: int,
) -> float:
    blocks, mask = build_night_tensors(window_index, embeddings, epoch_count)
    with torch.inference_mode():
        logit = sequence(blocks, mask, torch.zeros((1, 1), dtype=torch.float32))
    if not torch.isfinite(logit).all():
        raise FloatingPointError("Non-finite full-night CI logit")
    return float(logit.item())


def predict_psg(
    runtime: dict[str, Any],
    psg_path: Path,
    record_id: str,
    site_id: str,
) -> tuple[float, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="psg4ci_raw_") as directory:
        cache_path = Path(directory) / f"{record_id}.h5"
        online_features.build_psg_cache(psg_path, cache_path, record_id, site_id)
        indices, embeddings, epoch_count = encode_cache(
            cache_path, runtime["encoder"], int(runtime["encoder_batch_size"])
        )
    member_logits = [
        sequence_logit(sequence, indices, embeddings, epoch_count)
        for sequence in runtime["sequences"]
    ]
    logit = float(np.mean(member_logits))
    return logit, {
        "record_id": record_id,
        "complete_epoch_count": int(epoch_count),
        "eligible_epoch_count": int(len(indices)),
        "ensemble_members": int(len(member_logits)),
    }
