#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cache_builder import build_cache
from dataloader import (
    SequenceDataset,
    FixedDemographicsEncoder,
    collate_fn,
    default_cache_dir,
    default_demographics_path,
)
from psg4ci_model import PSG4CIModel


# ============================================================
# Defaults for final full-data training
# ============================================================
SEED = 42
BATCH_SIZE_TRAIN = 4
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
FREEZE_SLEEPFM = False
LR_SLEEPFM = 1e-5
LR_HEAD = 1e-4
SLEEPFM_CHUNK_BATCH = 8
GRAD_ACCUM_STEPS = 1
USE_DATA_PARALLEL = True
REQUIRE_CAISR = False

EPOCHS = 20

BUILD_CACHE = True
CACHE_OVERWRITE = False
CACHE_LIMIT_FILES = None
CACHE_TARGET_PSG_SFREQ = 128.0
CACHE_TARGET_ANN_SFREQ = None
CACHE_READ_VERBOSE = False
CACHE_FLUSH_EVERY = 5

DEFAULT_CACHE_NAME = "cache_h5_final_v1"
DEFAULT_OUTPUT_NAME = "final_train_run"


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=False)
        else:
            out[k] = v
    return out


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_model_state_dict(model):
    return unwrap_model(model).state_dict()


def maybe_build_cache(
    data_folder: Path | str,
    cache_dir: Path | str,
    build_cache_flag: bool = BUILD_CACHE,
    overwrite: bool = CACHE_OVERWRITE,
    limit_files: Optional[int] = CACHE_LIMIT_FILES,
    target_psg_sfreq: float = CACHE_TARGET_PSG_SFREQ,
    target_ann_sfreq: Optional[float] = CACHE_TARGET_ANN_SFREQ,
    read_verbose: bool = CACHE_READ_VERBOSE,
    flush_every: int = CACHE_FLUSH_EVERY,
    verbose: int = 1,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_h5 = list(cache_dir.glob("**/*.h5"))
    metadata_csv = cache_dir / "cache_metadata.csv"

    if not build_cache_flag:
        if verbose:
            print(f"[INFO] BUILD_CACHE=False, reuse cache under {cache_dir}")
        if metadata_csv.exists():
            try:
                return pd.read_csv(metadata_csv), cache_dir
            except Exception:
                return None, cache_dir
        return None, cache_dir

    if (len(existing_h5) > 0) and (not overwrite):
        if verbose:
            print(f"[INFO] cache already exists under {cache_dir} with {len(existing_h5)} H5 files; skip rebuild")
        if metadata_csv.exists():
            try:
                return pd.read_csv(metadata_csv), cache_dir
            except Exception:
                return None, cache_dir
        return None, cache_dir

    if verbose:
        print(f"[INFO] building cache into {cache_dir} ...")

    cache_df = build_cache(
        data_folder=data_folder,
        out_dir=cache_dir,
        overwrite=overwrite,
        limit_files=limit_files,
        target_psg_sfreq=target_psg_sfreq,
        target_ann_sfreq=target_ann_sfreq,
        read_verbose=read_verbose,
        flush_every=flush_every,
        metadata_csv=metadata_csv,
        verbose=verbose,
    )
    return cache_df, cache_dir


def build_full_dataset(
    data_folder: Path | str,
    cache_dir: Path | str,
    demog_path: Optional[Path | str] = None,
):
    data_folder = Path(data_folder)
    cache_dir = Path(cache_dir)
    if demog_path is None:
        demog_path = default_demographics_path(data_folder)
    demog_path = Path(demog_path)

    encoder = FixedDemographicsEncoder()
    ds = SequenceDataset(
        cache_dir=cache_dir,
        demog_path=demog_path,
        require_caisr=REQUIRE_CAISR,
        demographics_encoder=encoder,
        fit_demographics_encoder=True,
    )
    return ds, encoder, cache_dir, demog_path


def build_model(
    demo_dim: int,
    device: torch.device,
    sleepfm_repo_dir: Optional[Path | str] = None,
    sleepfm_ckpt_path: Optional[Path | str] = None,
):
    model = PSG4CIModel(
        demo_dim=demo_dim,
        freeze_sleepfm=FREEZE_SLEEPFM,
        dropout=DROPOUT,
        device_for_sleepfm=device,
        sleepfm_chunk_batch=SLEEPFM_CHUNK_BATCH,
        sleepfm_repo_dir=sleepfm_repo_dir,
        sleepfm_ckpt_path=sleepfm_ckpt_path,
    ).to(device)
    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def build_optimizer(model) -> torch.optim.Optimizer:
    core = unwrap_model(model)
    sleepfm_params = []
    head_params = []
    sleepfm_param_ids = {id(p) for p in core.sleepfm.parameters()}

    for p in core.parameters():
        if not p.requires_grad:
            continue
        if id(p) in sleepfm_param_ids:
            sleepfm_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
    if sleepfm_params:
        param_groups.append({"params": sleepfm_params, "lr": LR_SLEEPFM, "weight_decay": WEIGHT_DECAY})
    return torch.optim.AdamW(param_groups)


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs["ci_logits"], batch["y"].float())


def train_one_epoch(model, loader, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        outputs = model(batch)
        loss = compute_loss(outputs, batch)
        (loss / GRAD_ACCUM_STEPS).backward()

        if step % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.detach().cpu())
        n_steps += 1

    if n_steps > 0 and (n_steps % GRAD_ACCUM_STEPS != 0):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"loss": running_loss / max(n_steps, 1)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-data PSG4CI final training without validation split.")
    parser.add_argument("-d", "--data_folder", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--demog_path", type=str, default=None)
    parser.add_argument("--sleepfm_repo_dir", type=str, default=None)
    parser.add_argument("--sleepfm_ckpt_path", type=str, default=None,
                        help="Path to pretrained SleepFM backbone checkpoint.")
    parser.add_argument("--build_cache", action="store_true", default=BUILD_CACHE,
                        help="Build H5 cache before training.")
    parser.add_argument("--no_build_cache", action="store_false", dest="build_cache",
                        help="Reuse existing H5 cache and skip cache building.")
    parser.add_argument("--cache_overwrite", action="store_true", default=CACHE_OVERWRITE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size_train", type=int, default=BATCH_SIZE_TRAIN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(args=None):
    parser = build_argparser()
    args = parser.parse_args(args=args)

    set_seed(int(args.seed))
    device = get_device()

    data_folder = Path(args.data_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir is None:
        cache_dir = default_cache_dir(output_dir, cache_name=DEFAULT_CACHE_NAME)
    else:
        cache_dir = Path(args.cache_dir)

    cache_df, cache_dir = maybe_build_cache(
        data_folder=data_folder,
        cache_dir=cache_dir,
        build_cache_flag=bool(args.build_cache),
        overwrite=bool(args.cache_overwrite),
        limit_files=CACHE_LIMIT_FILES,
        verbose=int(args.verbose),
    )

    full_ds, encoder, cache_dir, demog_path = build_full_dataset(
        data_folder=data_folder,
        cache_dir=cache_dir,
        demog_path=args.demog_path,
    )

    train_loader = DataLoader(
        full_ds,
        batch_size=int(args.batch_size_train),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
    )

    model = build_model(
        demo_dim=encoder.output_dim,
        device=device,
        sleepfm_repo_dir=args.sleepfm_repo_dir,
        sleepfm_ckpt_path=args.sleepfm_ckpt_path,
    )
    optimizer = build_optimizer(model)

    if args.verbose:
        print(f"[INFO] device = {device}")
        print(f"[INFO] visible cuda count = {torch.cuda.device_count()}")
        print(f"[INFO] data parallel = {USE_DATA_PARALLEL}")
        print(f"[INFO] cache_dir = {cache_dir}")
        print(f"[INFO] demog_path = {demog_path}")
        print(f"[INFO] build_cache = {args.build_cache}")
        print(f"[INFO] sleepfm_ckpt_path = {args.sleepfm_ckpt_path}")
        print(f"[INFO] full dataset size = {len(full_ds)}")
        print("[INFO] labels:")
        print(full_ds.study_df["Cognitive_Impairment"].value_counts(dropna=False).to_string())
        if cache_df is not None and len(cache_df) > 0 and "status" in cache_df.columns:
            print("[INFO] cache status counts:")
            print(cache_df["status"].value_counts(dropna=False).to_string())

    history = []

    config = {
        "SEED": int(args.seed),
        "BATCH_SIZE_TRAIN": int(args.batch_size_train),
        "NUM_WORKERS": int(args.num_workers),
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT": DROPOUT,
        "FREEZE_SLEEPFM": FREEZE_SLEEPFM,
        "LR_SLEEPFM": LR_SLEEPFM,
        "LR_HEAD": LR_HEAD,
        "SLEEPFM_CHUNK_BATCH": SLEEPFM_CHUNK_BATCH,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "USE_DATA_PARALLEL": USE_DATA_PARALLEL,
        "REQUIRE_CAISR": REQUIRE_CAISR,
        "EPOCHS": int(args.epochs),
        "BUILD_CACHE": bool(args.build_cache),
        "CACHE_OVERWRITE": bool(args.cache_overwrite),
        "sleepfm_repo_dir": str(args.sleepfm_repo_dir) if args.sleepfm_repo_dir else None,
        "sleepfm_ckpt_path": str(args.sleepfm_ckpt_path) if args.sleepfm_ckpt_path else None,
        "cache_dir": str(cache_dir),
        "demog_path": str(demog_path),
        "device": str(device),
        "demo_dim": int(encoder.output_dim),
        "n_full": int(len(full_ds)),
    }

    for epoch in range(1, int(args.epochs) + 1):
        if args.verbose:
            print("\n" + "=" * 80)
            print(f"Final train epoch {epoch}")
            print("=" * 80)

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
        }
        history.append(row)

        final_ckpt = {
            "epoch": epoch,
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "demographics_encoder": {
                "numeric_cols": encoder.numeric_cols,
                "categorical_cols": encoder.categorical_cols,
                "category_levels": encoder.category_levels,
                "numeric_means": encoder.numeric_means,
                "numeric_stds": encoder.numeric_stds,
                "output_dim": encoder.output_dim,
            },
        }

        torch.save(final_ckpt, output_dir / "last_model.pt")
        pd.DataFrame(history).to_csv(output_dir / "final_train_history.csv", index=False)

        if args.verbose:
            print("[TRAIN]")
            print(f"  loss: {row['train_loss']:.6f}")

    torch.save(final_ckpt, output_dir / "final_model.pt")

    summary = {
        "final_epoch": int(history[-1]["epoch"]) if history else 0,
        "cache_dir": str(cache_dir),
        "demog_path": str(demog_path),
        "history_path": str(output_dir / "final_train_history.csv"),
        "final_model_path": str(output_dir / "final_model.pt"),
        "last_model_path": str(output_dir / "last_model.pt"),
    }
    (output_dir / "final_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "final_train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.verbose:
        print("\n[DONE] final full-data training finished.")
        print(f"[INFO] saved final model to: {output_dir / 'final_model.pt'}")

    return summary


if __name__ == "__main__":
    main()
