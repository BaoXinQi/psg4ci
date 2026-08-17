#!/usr/bin/env python3
"""v3 whole-night model with a training-only weak site adversary."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from sequence_ci_model import LocalFullNightTransformer


DOMAIN_REVERSAL_STRENGTH = 0.05


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx: object, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def reverse_gradient(values: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(values, float(strength))


class DomainAdversarialFullNightTransformer(LocalFullNightTransformer):
    def __init__(
        self,
        site_count: int,
        demographic_dimension: int = 1,
        d_model: int = 256,
        dropout: float = 0.15,
        reversal_strength: float = DOMAIN_REVERSAL_STRENGTH,
    ) -> None:
        super().__init__(
            demographic_dimension=demographic_dimension,
            use_demographics=False,
            d_model=d_model,
            dropout=dropout,
        )
        if site_count < 2:
            raise ValueError("Domain adversarial training requires at least two sites")
        self.reversal_strength = float(reversal_strength)
        rng_state = torch.random.get_rng_state()
        self.site_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, site_count),
        )
        torch.random.set_rng_state(rng_state)

    def representation(
        self, blocks: torch.Tensor, window_mask: torch.Tensor
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
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=blocks.device),
                block_mask,
            ],
            dim=1,
        )
        sequence = self.night_encoder(sequence, src_key_padding_mask=~sequence_mask)
        return self.output_norm(sequence[:, 0])

    def forward(
        self,
        blocks: torch.Tensor,
        window_mask: torch.Tensor,
        demographics: torch.Tensor,
        return_site: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del demographics
        representation = self.representation(blocks, window_mask)
        main_logit = self.head(representation).squeeze(-1)
        if return_site:
            site_logit = self.site_head(
                reverse_gradient(representation, self.reversal_strength)
            )
            return main_logit, site_logit
        return main_logit
