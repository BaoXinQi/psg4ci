#!/usr/bin/env python3
"""All-modality 30-second PSG encoder used by the E0/E1 pilot."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


MODALITY_CHANNELS = {
    "eeg": 6,
    "eog": 2,
    "ecg": 1,
    "resp": 7,
    "spo2": 1,
    "emg": 3,
}
STEM_WIDTHS = {
    "eeg": (24, 40, 56),
    "eog": (16, 28, 40),
    "ecg": (16, 28, 40),
    "resp": (20, 32, 48),
    "spo2": (12, 20, 24),
    "emg": (16, 28, 40),
}


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            group_norm(channels),
            nn.GELU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal + self.network(signal)


class ModalityStem(nn.Module):
    def __init__(self, input_channels: int, widths: tuple[int, int, int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_channels
        for output, kernel, stride in zip(widths, (17, 11, 7), (4, 4, 2)):
            layers.extend(
                [
                    nn.Conv1d(
                        current,
                        output,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    group_norm(output),
                    nn.GELU(),
                ]
            )
            current = output
        self.network = nn.Sequential(*layers)
        self.residual = ResidualTemporalBlock(widths[-1], dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.attention = nn.Linear(widths[-1], 1)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.pool(self.residual(self.network(signal))).transpose(1, 2)
        weights = torch.softmax(self.attention(tokens).squeeze(-1), dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1), tokens


class DomainRobustPsgEncoder(nn.Module):
    embedding_dimension = 192
    projection_dimension = 128

    def __init__(self) -> None:
        super().__init__()
        self.stems = nn.ModuleDict(
            {
                name: ModalityStem(MODALITY_CHANNELS[name], STEM_WIDTHS[name])
                for name in MODALITY_CHANNELS
            }
        )
        stem_dimension = sum(widths[-1] for widths in STEM_WIDTHS.values())
        availability_dimension = sum(MODALITY_CHANNELS.values())
        self.fusion = nn.Sequential(
            nn.Linear(stem_dimension + availability_dimension, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(320, self.embedding_dimension),
            nn.LayerNorm(self.embedding_dimension),
        )
        self.stage_head = nn.Linear(self.embedding_dimension, 5)
        self.event_head = nn.Linear(self.embedding_dimension, 3)
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dimension, 256),
            nn.GELU(),
            nn.Linear(256, self.projection_dimension),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.projection_dimension, 256),
            nn.GELU(),
            nn.Linear(256, self.projection_dimension),
        )
        self.modality_projectors = nn.ModuleDict(
            {
                name: nn.Linear(STEM_WIDTHS[name][-1], 64)
                for name in MODALITY_CHANNELS
            }
        )
        self.modality_predictors = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(64, 96), nn.GELU(), nn.Linear(96, 64))
                for name in MODALITY_CHANNELS
            }
        )
        self.token_contexts = nn.ModuleDict(
            {
                name: nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=STEM_WIDTHS[name][-1],
                        nhead=4,
                        dim_feedforward=2 * STEM_WIDTHS[name][-1],
                        dropout=0.10,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    ),
                    num_layers=1,
                )
                for name in MODALITY_CHANNELS
            }
        )
        self.token_projectors = nn.ModuleDict(
            {
                name: nn.Linear(STEM_WIDTHS[name][-1], 64)
                for name in MODALITY_CHANNELS
            }
        )
        self.token_predictors = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(64, 96), nn.GELU(), nn.Linear(96, 64))
                for name in MODALITY_CHANNELS
            }
        )

    def encode(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        modality_vectors: dict[str, torch.Tensor] = {}
        modality_tokens: dict[str, torch.Tensor] = {}
        for name, stem in self.stems.items():
            vector, tokens = stem(batch[name].float())
            modality_vectors[name] = vector
            modality_tokens[name] = tokens
        availability = torch.cat(
            [batch[f"{name}_mask"].float() for name in MODALITY_CHANNELS], dim=1
        )
        embedding = self.fusion(
            torch.cat([*modality_vectors.values(), availability], dim=1)
        )
        return embedding, modality_vectors, modality_tokens

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, object]:
        embedding, modality_vectors, modality_tokens = self.encode(batch)
        projections = {
            name: self.modality_projectors[name](vector)
            for name, vector in modality_vectors.items()
        }
        token_projections = {
            name: self.token_projectors[name](self.token_contexts[name](tokens))
            for name, tokens in modality_tokens.items()
        }
        return {
            "embedding": embedding,
            "stage_logits": self.stage_head(embedding),
            "event_logits": self.event_head(embedding),
            "projection": self.projector(embedding),
            "modality_projections": projections,
            "token_projections": token_projections,
        }

    def predict_projection(self, projection: torch.Tensor) -> torch.Tensor:
        return self.predictor(projection)

    def predict_modality(
        self, name: str, projection: torch.Tensor
    ) -> torch.Tensor:
        return self.modality_predictors[name](projection)

    def predict_tokens(
        self, name: str, projection: torch.Tensor
    ) -> torch.Tensor:
        return self.token_predictors[name](projection)
