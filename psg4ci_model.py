#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

import torch
import torch.nn as nn

SLEEPFM_CONFIG = {
    'patch_size': 640,
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 6,
    'pooling_head': 4,
    'dropout': 0.1,
    'max_seq_length': 128,
}

MODALITIES = ['bas', 'resp', 'ekg', 'emg']
EXPECTED_TOKEN_LEN = 640
CAISR_DIM = 11
EMBED_DIM = 128


def resolve_sleepfm_repo_dir(repo_dir: Optional[Path | str] = None) -> Path:
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parent / 'sleepfm_core'
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        raise FileNotFoundError(f"SleepFM repo directory not found: {repo_dir}")
    return repo_dir


def prepare_sleepfm_import(repo_dir: Path) -> None:
    sleepfm_dir = repo_dir / 'sleepfm'
    for p in [str(repo_dir), str(sleepfm_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def build_sleepfm_backbone(device: torch.device, repo_dir: Optional[Path | str] = None) -> nn.Module:
    repo_dir = resolve_sleepfm_repo_dir(repo_dir=repo_dir)
    prepare_sleepfm_import(repo_dir)
    from sleepfm.models.models import SetTransformer  # type: ignore

    model = SetTransformer(
        in_channels=1,
        patch_size=SLEEPFM_CONFIG['patch_size'],
        embed_dim=SLEEPFM_CONFIG['embed_dim'],
        num_heads=SLEEPFM_CONFIG['num_heads'],
        num_layers=SLEEPFM_CONFIG['num_layers'],
        pooling_head=SLEEPFM_CONFIG['pooling_head'],
        dropout=SLEEPFM_CONFIG['dropout'],
        max_seq_length=SLEEPFM_CONFIG['max_seq_length'],
    )
    model.to(device)
    return model


def maybe_load_sleepfm_base_weights(model: nn.Module, ckpt_path: Optional[Path | str] = None) -> None:
    if ckpt_path is None:
        return
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SleepFM base checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)


def load_sleepfm_backbone(
    device: torch.device,
    repo_dir: Optional[Path | str] = None,
    ckpt_path: Optional[Path | str] = None,
) -> nn.Module:
    model = build_sleepfm_backbone(device=device, repo_dir=repo_dir)
    maybe_load_sleepfm_base_weights(model, ckpt_path=ckpt_path)
    return model


def masked_mean_pool_1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)


class CAISRSequenceEncoder(nn.Module):
    def __init__(self, in_dim: int = CAISR_DIM, embed_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        b, n, t, f = x.shape
        x = x.reshape(b * n, t, f)
        mask = token_mask.reshape(b * n, t).bool()
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = x * mask.unsqueeze(-1).float()
        return x.reshape(b, n, t, -1)


class PSG4CIModel(nn.Module):
    def __init__(
        self,
        demo_dim: int,
        freeze_sleepfm: bool = True,
        dropout: float = 0.1,
        device_for_sleepfm: Optional[torch.device] = None,
        sleepfm_chunk_batch: int = 16,
        sleepfm_repo_dir: Optional[Path | str] = None,
        sleepfm_ckpt_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__()
        if device_for_sleepfm is None:
            device_for_sleepfm = torch.device('cpu')

        self.sleepfm = load_sleepfm_backbone(
            device=device_for_sleepfm,
            repo_dir=sleepfm_repo_dir,
            ckpt_path=sleepfm_ckpt_path,
        )

        if freeze_sleepfm:
            for p in self.sleepfm.parameters():
                p.requires_grad = False
            self.sleepfm.eval()

        self.sleepfm_chunk_batch = int(sleepfm_chunk_batch)

        self.caisr_encoder = CAISRSequenceEncoder(CAISR_DIM, EMBED_DIM, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(EMBED_DIM),
        )

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=EMBED_DIM // 2,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.demo_branch = nn.Sequential(
            nn.LayerNorm(demo_dim),
            nn.Linear(demo_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _run_sleepfm_in_chunks(self, x_valid: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
        outs = []
        k = x_valid.shape[0]
        step = max(1, self.sleepfm_chunk_batch)

        if (not self.training) or all(not p.requires_grad for p in self.sleepfm.parameters()):
            self.sleepfm.eval()
            with torch.no_grad():
                for s in range(0, k, step):
                    e = min(s + step, k)
                    _, seq_part = self.sleepfm(x_valid[s:e], mask_valid[s:e])
                    outs.append(seq_part)
        else:
            for s in range(0, k, step):
                e = min(s + step, k)
                _, seq_part = self.sleepfm(x_valid[s:e], mask_valid[s:e])
                outs.append(seq_part)

        return torch.cat(outs, dim=0)

    def _modality_forward(self, x_mod: torch.Tensor, mod_mask: torch.Tensor) -> torch.Tensor:
        b, n, t, c, l = x_mod.shape
        if l != EXPECTED_TOKEN_LEN:
            raise ValueError(f'Expected token len {EXPECTED_TOKEN_LEN}, got {l}')

        x = x_mod.reshape(b * n, t, c, l).permute(0, 2, 1, 3).contiguous().reshape(b * n, c, t * l)
        ch_present = mod_mask[:, None, :].expand(b, n, c).reshape(b * n, c).bool()

        out = torch.zeros((b * n, t, EMBED_DIM), device=x.device, dtype=x.dtype)
        valid_rows = (ch_present.sum(dim=1) > 0)
        if valid_rows.any():
            x_valid = x[valid_rows]
            mask_valid = ~ch_present[valid_rows]
            out_valid = self._run_sleepfm_in_chunks(x_valid, mask_valid)
            out[valid_rows] = out_valid

        return out.reshape(b, n, t, EMBED_DIM)

    def encode_psg_sequence(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        b = batch['token_mask'].shape[0]
        n = batch['token_mask'].shape[1]
        t = batch['token_mask'].shape[2]
        device = batch['token_mask'].device

        z_sum = torch.zeros((b, n, t, EMBED_DIM), device=device, dtype=torch.float32)
        mod_presence = []

        for mod in MODALITIES:
            z_mod = self._modality_forward(batch[f'{mod}_chunks'], batch[f'{mod}_channel_mask'])
            z_mod = torch.nan_to_num(z_mod, nan=0.0, posinf=0.0, neginf=0.0)

            present = (batch[f'{mod}_channel_mask'].sum(dim=1) > 0)
            mod_presence.append(present)
            z_sum = z_sum + z_mod * present[:, None, None, None].float()

        mod_mask = torch.stack(mod_presence, dim=1)
        denom = mod_mask.sum(dim=1).float().clamp_min(1.0)
        z_psg = z_sum / denom[:, None, None, None]
        return z_psg, mod_mask

    def encode_caisr_sequence(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_mask = batch['token_mask'].bool()
        z_ann = self.caisr_encoder(batch['caisr_chunks'].float(), token_mask)
        z_ann = torch.nan_to_num(z_ann, nan=0.0, posinf=0.0, neginf=0.0)

        sample_has_caisr = batch.get('has_caisr', None)
        if sample_has_caisr is not None:
            sample_has_caisr = sample_has_caisr.view(-1, 1, 1, 1).float()
            z_ann = z_ann * sample_has_caisr

        channel_mask = batch.get('caisr_channel_mask', None)
        if channel_mask is not None:
            channel_has_signal = (channel_mask.sum(dim=1) > 0).view(-1, 1, 1, 1).float()
            z_ann = z_ann * channel_has_signal

        return z_ann

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_mask = batch['token_mask'].bool()
        chunk_mask = batch['chunk_mask'].bool()

        z_psg, mod_mask = self.encode_psg_sequence(batch)
        z_psg = torch.nan_to_num(z_psg, nan=0.0, posinf=0.0, neginf=0.0)
        z_ann = self.encode_caisr_sequence(batch)
        z = self.fusion(torch.cat([z_psg, z_ann], dim=-1))

        b, n, t, d = z.shape
        z = z.reshape(b, n * t, d)
        seq_mask = token_mask.reshape(b, n * t)

        lengths = seq_mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        z_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=n * t)

        z_file = masked_mean_pool_1d(z_seq, seq_mask)
        z_demo = self.demo_branch(torch.nan_to_num(batch['demo_x'].float(), nan=0.0, posinf=0.0, neginf=0.0))
        logits = self.head(torch.cat([z_file, z_demo], dim=-1)).squeeze(-1)

        return {
            'z_psg_seq': z_psg,
            'z_ann_seq': z_ann,
            'z_fused_seq': z_seq,
            'z_file': z_file,
            'demo': z_demo,
            'ci_logits': logits,
            'seq_mask': seq_mask,
            'mod_mask': mod_mask,
            'chunk_mask': chunk_mask,
        }


def compute_ci_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    y = batch['y'].float()
    return nn.functional.binary_cross_entropy_with_logits(outputs['ci_logits'], y)


if __name__ == '__main__':
    print("psg4ci_model.py written.")
