#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cache_builder import build_cache
from dataloader import (
    SequenceDataset,
    FixedDemographicsEncoder,
    collate_fn,
    default_demographics_path,
    default_cache_dir,
)
from psg4ci_model import PSG4CIModel

SEED = 42
BATCH_SIZE_TRAIN = 1 # 4
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
FREEZE_SLEEPFM = True # False
LR_SLEEPFM = 1e-5
LR_HEAD = 1e-4
SLEEPFM_CHUNK_BATCH = 2 # 8
GRAD_ACCUM_STEPS = 1
USE_DATA_PARALLEL = False # True
REQUIRE_CAISR = False
SUBMISSION_FINETUNE_EPOCHS = 2

CACHE_OVERWRITE = False
CACHE_LIMIT_FILES = None
CACHE_TARGET_PSG_SFREQ = 128.0
CACHE_TARGET_ANN_SFREQ = None
CACHE_READ_VERBOSE = False
CACHE_FLUSH_EVERY = 5


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
    sleepfm_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    sleepfm_param_ids = {id(p) for p in core.sleepfm.parameters()}

    for p in core.parameters():
        if not p.requires_grad:
            continue
        if id(p) in sleepfm_param_ids:
            sleepfm_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
    if len(sleepfm_params) > 0:
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

    if n_steps % GRAD_ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"loss": running_loss / max(n_steps, 1)}


def maybe_build_cache(
    data_folder: Path | str,
    model_folder: Path | str,
    cache_dir: Optional[Path | str] = None,
    overwrite: bool = CACHE_OVERWRITE,
    limit_files: Optional[int] = CACHE_LIMIT_FILES,
    target_psg_sfreq: float = CACHE_TARGET_PSG_SFREQ,
    target_ann_sfreq: Optional[float] = CACHE_TARGET_ANN_SFREQ,
    read_verbose: bool = CACHE_READ_VERBOSE,
    flush_every: int = CACHE_FLUSH_EVERY,
    verbose: int = 1,
):
    model_folder = Path(model_folder)
    if cache_dir is None:
        cache_dir = default_cache_dir(model_folder)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_h5 = list(cache_dir.glob("**/*.h5"))
    metadata_csv = cache_dir / "cache_metadata.csv"

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


def build_dataset(
    data_folder: Path | str,
    model_folder: Path | str,
    cache_dir: Optional[Path | str] = None,
    demog_path: Optional[Path | str] = None,
):
    data_folder = Path(data_folder)
    model_folder = Path(model_folder)

    if demog_path is None:
        demog_path = default_demographics_path(data_folder)
    demog_path = Path(demog_path)

    if cache_dir is None:
        cache_dir = default_cache_dir(model_folder)
    cache_dir = Path(cache_dir)

    encoder = FixedDemographicsEncoder()
    ds = SequenceDataset(
        cache_dir=cache_dir,
        demog_path=demog_path,
        require_caisr=REQUIRE_CAISR,
        demographics_encoder=encoder,
        fit_demographics_encoder=True,
    )
    return ds, encoder, cache_dir, demog_path


def load_resume_weights_if_available(model, optimizer, resume_checkpoint: Optional[Path | str]):
    if resume_checkpoint is None:
        return None
    resume_checkpoint = Path(resume_checkpoint)
    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    ckpt = torch.load(resume_checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"]

    # Handle checkpoints saved from wrapped models, e.g. keys like:
    #   base.sleepfm....
    # or DataParallel keys like:
    #   module.base.sleepfm....
    cleaned_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("base."):
            nk = nk[len("base."):]
        cleaned_state[nk] = v

    unwrap_model(model).load_state_dict(cleaned_state, strict=True)

    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    return ckpt


def train_submission(
    data_folder: Path | str,
    model_folder: Path | str,
    resume_checkpoint: Optional[Path | str] = None,
    cache_dir: Optional[Path | str] = None,
    demog_path: Optional[Path | str] = None,
    sleepfm_repo_dir: Optional[Path | str] = None,
    sleepfm_ckpt_path: Optional[Path | str] = None,
    finetune_epochs: int = SUBMISSION_FINETUNE_EPOCHS,
    cache_overwrite: bool = CACHE_OVERWRITE,
    cache_limit_files: Optional[int] = CACHE_LIMIT_FILES,
    verbose: int = 1,
):
    set_seed(SEED)
    device = get_device()

    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    cache_df, cache_dir = maybe_build_cache(
        data_folder=data_folder,
        model_folder=model_folder,
        cache_dir=cache_dir,
        overwrite=cache_overwrite,
        limit_files=cache_limit_files,
        verbose=verbose,
    )

    ds, encoder, cache_dir, demog_path = build_dataset(
        data_folder=data_folder,
        model_folder=model_folder,
        cache_dir=cache_dir,
        demog_path=demog_path,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = build_model(
        demo_dim=encoder.output_dim,
        device=device,
        sleepfm_repo_dir=sleepfm_repo_dir,
        sleepfm_ckpt_path=sleepfm_ckpt_path,
    )
    optimizer = build_optimizer(model)
    resume_meta = load_resume_weights_if_available(model, optimizer, resume_checkpoint)

    if verbose:
        print(f"[INFO] device = {device}")
        print(f"[INFO] visible cuda count = {torch.cuda.device_count()}")
        print(f"[INFO] data parallel = {USE_DATA_PARALLEL}")
        print(f"[INFO] dataset size = {len(ds)}")
        print(f"[INFO] demographics dim = {encoder.output_dim}")
        print(f"[INFO] cache_dir = {cache_dir}")
        print(f"[INFO] demog_path = {demog_path}")
        print(f"[INFO] resume_checkpoint = {resume_checkpoint}")
        if cache_df is not None and len(cache_df) > 0 and "status" in cache_df.columns:
            print("[INFO] cache status counts:")
            print(cache_df["status"].value_counts(dropna=False).to_string())
        if hasattr(ds, "study_df"):
            print("[INFO] labels:")
            print(ds.study_df["Cognitive_Impairment"].value_counts(dropna=False).to_string())
            if "has_caisr" in ds.study_df.columns:
                print("[INFO] has_caisr:")
                print(ds.study_df["has_caisr"].value_counts(dropna=False).to_string())

    history = []
    global_start_epoch = 1
    if resume_meta is not None and "epoch" in resume_meta:
        global_start_epoch = int(resume_meta["epoch"]) + 1

    for submission_epoch in range(1, int(finetune_epochs) + 1):
        global_epoch = global_start_epoch + submission_epoch - 1
        if verbose:
            print("\n" + "=" * 80)
            print(f"Submission epoch {submission_epoch}")
            print("=" * 80)

        train_metrics = train_one_epoch(model, loader, optimizer, device)
        history.append({
            "epoch": submission_epoch,
            "submission_epoch": submission_epoch,
            "global_epoch": global_epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
        })

        if verbose:
            print("[TRAIN]")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.6f}")

    config = {
        "SEED": SEED,
        "BATCH_SIZE_TRAIN": BATCH_SIZE_TRAIN,
        "NUM_WORKERS": NUM_WORKERS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT": DROPOUT,
        "FREEZE_SLEEPFM": FREEZE_SLEEPFM,
        "LR_SLEEPFM": LR_SLEEPFM,
        "LR_HEAD": LR_HEAD,
        "SLEEPFM_CHUNK_BATCH": SLEEPFM_CHUNK_BATCH,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "USE_DATA_PARALLEL": USE_DATA_PARALLEL,
        "REQUIRE_CAISR": REQUIRE_CAISR,
        "SUBMISSION_FINETUNE_EPOCHS": int(finetune_epochs),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "sleepfm_repo_dir": str(sleepfm_repo_dir) if sleepfm_repo_dir is not None else None,
        "sleepfm_ckpt_path": str(sleepfm_ckpt_path) if sleepfm_ckpt_path is not None else None,
        "device": str(device),
        "cuda_count": int(torch.cuda.device_count()),
        "demo_dim": int(encoder.output_dim),
        "n_train_full": int(len(ds)),
        "cache_dir": str(cache_dir),
        "demog_path": str(demog_path),
        "cache_overwrite": bool(cache_overwrite),
        "cache_limit_files": None if cache_limit_files is None else int(cache_limit_files),
    }

    pd.DataFrame(history).to_csv(model_folder / "submission_train_history.csv", index=False)
    (model_folder / "submission_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    ckpt = {
        "epoch": history[-1]["global_epoch"] if len(history) > 0 else 0,
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
    torch.save(ckpt, model_folder / "submission_model.pt")

    if verbose:
        print("\n[DONE] submission training finished.")
        print(f"[INFO] saved model to: {model_folder / 'submission_model.pt'}")

    return {
        "model_path": model_folder / "submission_model.pt",
        "config_path": model_folder / "submission_config.json",
        "history_path": model_folder / "submission_train_history.csv",
        "cache_dir": cache_dir,
        "demog_path": demog_path,
    }


if __name__ == "__main__":
    print("submission_train.py updated.")
