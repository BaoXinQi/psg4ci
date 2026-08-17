#!/usr/bin/env python3
"""Fixed local-block and full-night Transformer for frozen E1 embeddings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import BatchSampler, Dataset


EMBEDDING_DIMENSION = 192
WINDOWS_PER_BLOCK = 10
MAX_BLOCKS = 256


class FullNightEmbeddingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, demographics: np.ndarray) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.demographics = np.asarray(demographics, dtype=np.float32)
        if len(self.frame) != len(self.demographics):
            raise ValueError("Frame and demographic matrix lengths differ")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with h5py.File(str(row["shard_path"]), "r") as handle:
            group = handle[str(row["group_key"])]
            embedding = np.asarray(group["embedding"], dtype=np.float32)
            window_index = np.asarray(group["window_index"], dtype=np.int64)
        epoch_count = max(int(row["complete_epoch_count"]), 1)
        block_count = min(max(math.ceil(epoch_count / WINDOWS_PER_BLOCK), 1), MAX_BLOCKS)
        blocks = np.zeros(
            (block_count, WINDOWS_PER_BLOCK, EMBEDDING_DIMENSION), dtype=np.float32
        )
        mask = np.zeros((block_count, WINDOWS_PER_BLOCK), dtype=bool)
        valid = window_index < block_count * WINDOWS_PER_BLOCK
        if np.any(valid):
            selected_index = window_index[valid]
            block = selected_index // WINDOWS_PER_BLOCK
            offset = selected_index % WINDOWS_PER_BLOCK
            blocks[block, offset] = embedding[valid]
            mask[block, offset] = True
        if not np.isfinite(blocks).all():
            raise FloatingPointError(f"Non-finite embedding sequence: {row['record_id']}")
        return {
            "blocks": torch.from_numpy(blocks),
            "window_mask": torch.from_numpy(mask),
            "demographics": torch.from_numpy(self.demographics[index]),
            "label": torch.tensor(float(row["_label"]), dtype=torch.float32),
            "age": torch.tensor(float(row["_age"]), dtype=torch.float32),
            "site_id": str(row["SiteID"]),
            "record_id": str(row["record_id"]),
            "row_index": int(index),
        }


def collate_full_nights(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    maximum_blocks = max(int(row["blocks"].shape[0]) for row in rows)
    batch_size = len(rows)
    blocks = torch.zeros(
        batch_size,
        maximum_blocks,
        WINDOWS_PER_BLOCK,
        EMBEDDING_DIMENSION,
        dtype=torch.float32,
    )
    window_mask = torch.zeros(
        batch_size, maximum_blocks, WINDOWS_PER_BLOCK, dtype=torch.bool
    )
    for index, row in enumerate(rows):
        count = int(row["blocks"].shape[0])
        blocks[index, :count] = row["blocks"]
        window_mask[index, :count] = row["window_mask"]
    return {
        "blocks": blocks,
        "window_mask": window_mask,
        "demographics": torch.stack([row["demographics"] for row in rows]),
        "label": torch.stack([row["label"] for row in rows]),
        "age": torch.stack([row["age"] for row in rows]),
        "site_id": [str(row["site_id"]) for row in rows],
        "record_id": [str(row["record_id"]) for row in rows],
        "row_index": torch.as_tensor([int(row["row_index"]) for row in rows]),
    }


class NaturalWithSiteBalancedBatchSampler(BatchSampler):
    """All natural samples plus one site-balanced batch after four natural batches."""

    def __init__(
        self,
        sites: Sequence[str],
        batch_size: int,
        seed: int,
        balanced_fraction: float = 0.20,
    ) -> None:
        if balanced_fraction != 0.20:
            raise ValueError("The frozen protocol requires balanced_fraction=0.20")
        self.sites = np.asarray(sites).astype(str)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.site_indices = {
            site: np.flatnonzero(self.sites == site) for site in sorted(set(self.sites))
        }
        self.natural_batches = math.ceil(len(self.sites) / self.batch_size)
        self.total_batches = math.ceil(self.natural_batches / 0.80)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1009)
        natural = rng.permutation(len(self.sites)).tolist()
        natural_position = 0
        site_names = list(self.site_indices)
        for batch_index in range(self.total_batches):
            if (batch_index + 1) % 5 == 0 and len(site_names) > 1:
                base = self.batch_size // len(site_names)
                remainder = self.batch_size % len(site_names)
                batch: list[int] = []
                for site_index, site in enumerate(site_names):
                    count = base + int(site_index < remainder)
                    values = rng.choice(self.site_indices[site], size=count, replace=True)
                    batch.extend(int(value) for value in values)
                rng.shuffle(batch)
                yield batch
                continue
            batch = natural[natural_position : natural_position + self.batch_size]
            natural_position += len(batch)
            if len(batch) < self.batch_size:
                extra = rng.choice(len(self.sites), size=self.batch_size - len(batch), replace=True)
                batch.extend(int(value) for value in extra)
            yield batch


class LocalFullNightTransformer(nn.Module):
    def __init__(
        self,
        demographic_dimension: int,
        use_demographics: bool,
        d_model: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.use_demographics = bool(use_demographics)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(EMBEDDING_DIMENSION),
            nn.Linear(EMBEDDING_DIMENSION, d_model),
        )
        self.local_position = nn.Parameter(torch.zeros(1, WINDOWS_PER_BLOCK, d_model))
        local_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.night_position = nn.Parameter(torch.zeros(1, MAX_BLOCKS + 1, d_model))
        night_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.night_encoder = nn.TransformerEncoder(night_layer, num_layers=3)
        self.output_norm = nn.LayerNorm(d_model)
        if self.use_demographics:
            self.demographic_encoder = nn.Sequential(
                nn.Linear(demographic_dimension, 32),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input = d_model + 32
        else:
            self.demographic_encoder = None
            head_input = d_model
        self.head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.local_position, std=0.02)
        nn.init.normal_(self.night_position, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        blocks: torch.Tensor,
        window_mask: torch.Tensor,
        demographics: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, block_count, window_count, _ = blocks.shape
        flattened = blocks.reshape(batch_size * block_count, window_count, -1)
        flattened_mask = window_mask.reshape(batch_size * block_count, window_count)
        block_valid = flattened_mask.any(dim=1)
        block_vectors = torch.zeros(
            batch_size * block_count,
            self.cls_token.shape[-1],
            device=blocks.device,
            dtype=blocks.dtype,
        )
        if torch.any(block_valid):
            selected = self.input_projection(flattened[block_valid]) + self.local_position
            selected_mask = flattened_mask[block_valid]
            selected = self.local_encoder(
                selected, src_key_padding_mask=~selected_mask
            )
            denominator = selected_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (selected * selected_mask.unsqueeze(-1)).sum(dim=1) / denominator
            block_vectors = block_vectors.index_copy(
                0, torch.nonzero(block_valid, as_tuple=False).flatten(), pooled
            )
        block_vectors = block_vectors.reshape(batch_size, block_count, -1)
        block_mask = window_mask.any(dim=2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, block_vectors], dim=1)
        sequence = sequence + self.night_position[:, : block_count + 1]
        sequence_mask = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=blocks.device), block_mask],
            dim=1,
        )
        sequence = self.night_encoder(
            sequence, src_key_padding_mask=~sequence_mask
        )
        representation = self.output_norm(sequence[:, 0])
        if self.use_demographics:
            representation = torch.cat(
                [representation, self.demographic_encoder(demographics)], dim=1
            )
        return self.head(representation).squeeze(-1)
