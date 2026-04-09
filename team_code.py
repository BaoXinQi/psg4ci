#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="Channels contain different highpass filters.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Channels contain different lowpass filters.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool.*",
    category=UserWarning,
)

import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import torch

from submission_train import train_submission
from psg4ci_model import PSG4CIModel
from dataloader import (
    FixedDemographicsEncoder,
    CAISR_CHANNELS,
    MODALITIES,
    PSG_SAMPLES_PER_TOKEN,
    ANN_SAMPLES_PER_TOKEN,
    load_demographics_table,
    ann_to_token_level,
    select_annotation_channels,
)
from channel_mapper import group_channels_for_sleepfm
from cache_builder import (
    infer_train_root,
    build_index_by_basename,
    find_matching_annotation_files,
    read_and_optionally_resample,
)

# Optional import from the official template environment.
try:
    from helper_code import *  # noqa: F401,F403
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESUME_CHECKPOINT = SCRIPT_DIR / "psg4ci.pt"
DEFAULT_SLEEPFM_REPO_DIR = SCRIPT_DIR / "sleepfm_core"
DEFAULT_SUBMISSION_MODEL = "submission_model.pt"
DEFAULT_THRESHOLD = 0.5
TARGET_PSG_SFREQ = 128.0
TARGET_ANN_SFREQ = 2.0


################################################################################
#
# Required functions. Do not change the arguments.
#
################################################################################

def train_model(data_folder, model_folder, verbose):
    data_folder = Path(data_folder)
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_RESUME_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Expected pretrained checkpoint at {DEFAULT_RESUME_CHECKPOINT}. "
            "Please place your renamed epoch18 checkpoint there."
        )

    result = train_submission(
        data_folder=data_folder,
        model_folder=model_folder,
        resume_checkpoint=DEFAULT_RESUME_CHECKPOINT,
        sleepfm_repo_dir=DEFAULT_SLEEPFM_REPO_DIR,
        sleepfm_ckpt_path=None,
        finetune_epochs=2,
        cache_overwrite=False,
        cache_limit_files=None,
        verbose=int(verbose),
    )

    metadata = {
        "submission_model": str(result["model_path"]),
        "submission_config": str(result["config_path"]),
        "submission_history": str(result["history_path"]),
        "cache_dir": str(result["cache_dir"]),
        "demog_path": str(result["demog_path"]),
        "resume_checkpoint": str(DEFAULT_RESUME_CHECKPOINT),
        "sleepfm_repo_dir": str(DEFAULT_SLEEPFM_REPO_DIR),
    }
    (model_folder / "team_metadata.json").write_text(
        pd.Series(metadata).to_json(), encoding="utf-8"
    )


def load_model(model_folder, verbose):
    model_folder = Path(model_folder)
    model_path = model_folder / DEFAULT_SUBMISSION_MODEL

    if not model_path.exists():
        raise FileNotFoundError(f"Trained submission model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    enc_info = ckpt["demographics_encoder"]

    encoder = FixedDemographicsEncoder(
        numeric_cols=enc_info["numeric_cols"],
        categorical_cols=enc_info["categorical_cols"],
        category_levels=enc_info["category_levels"],
    )
    encoder.numeric_means = {k: float(v) for k, v in enc_info["numeric_means"].items()}
    encoder.numeric_stds = {k: float(v) for k, v in enc_info["numeric_stds"].items()}
    encoder.output_dim = int(enc_info["output_dim"])
    encoder.is_fitted = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PSG4CIModel(
        demo_dim=encoder.output_dim,
        freeze_sleepfm=True,
        dropout=float(ckpt["config"].get("DROPOUT", 0.1)),
        device_for_sleepfm=device,
        sleepfm_chunk_batch=int(ckpt["config"].get("SLEEPFM_CHUNK_BATCH", 8)),
        sleepfm_repo_dir=DEFAULT_SLEEPFM_REPO_DIR,
        sleepfm_ckpt_path=None,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    state = {
        "model": model,
        "device": device,
        "encoder": encoder,
        "threshold": float(ckpt["config"].get("threshold", DEFAULT_THRESHOLD)),
        "demographics_df": None,
        "basename_index": None,
        "train_root": None,
    }

    if verbose:
        print(f"Loaded model from {model_path} on {device}")

    return state


def run_model(model, record, data_folder, verbose):
    state = model
    net = state["model"]
    device = state["device"]
    encoder = state["encoder"]
    threshold = state.get("threshold", DEFAULT_THRESHOLD)

    record_info = _parse_record(record)
    patient_id = record_info["patient_id"]
    site_id = record_info["site_id"]
    session_id = record_info["session_id"]
    file_name = f"{patient_id}_ses-{session_id}.edf"

    data_folder = Path(data_folder)
    train_root = infer_train_root(data_folder)

    if state["train_root"] != str(train_root) or state["demographics_df"] is None:
        demog_path = _default_demographics_path(data_folder)
        state["demographics_df"] = load_demographics_table(demog_path)
        state["basename_index"] = build_index_by_basename(train_root)
        state["train_root"] = str(train_root)

    demog_df = state["demographics_df"]
    basename_index = state["basename_index"]

    row = _get_demographics_row(demog_df, patient_id, session_id, file_name)
    batch = _build_single_record_batch(
        row=row,
        site_id=site_id,
        file_name=file_name,
        train_root=train_root,
        basename_index=basename_index,
        encoder=encoder,
    )

    batch = _move_batch(batch, device) 
    with torch.no_grad():
        outputs = net(batch)
        # For the retrained model, sigmoid(logit) directly represents
        # P(Cognitive_Impairment=True) in the official Challenge semantics.
        prob = torch.sigmoid(outputs["ci_logits"])[0].item()

    binary = bool(prob >= threshold)
    return binary, float(prob)


################################################################################
#
# Internal helpers
#
################################################################################

def _default_demographics_path(data_folder: Path) -> Path:
    candidates = [
        data_folder / "training_set" / "demographics.csv",
        data_folder / "demographics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find demographics.csv under {data_folder}")


def _parse_record(record: Dict[str, Any]) -> Dict[str, str]:
    patient_id = _get_any(record, ["BidsFolder", "bids_folder", "patient_id", "PatientID"])
    site_id = _get_any(record, ["SiteID", "site_id", "site"])
    session_id = _get_any(record, ["SessionID", "session_id", "session"])
    if patient_id is None or site_id is None or session_id is None:
        raise ValueError(f"Could not parse required fields from record: keys={list(record.keys())}")
    return {
        "patient_id": str(patient_id),
        "site_id": str(site_id),
        "session_id": str(session_id),
    }


def _get_any(d: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _get_demographics_row(df: pd.DataFrame, patient_id: str, session_id: str, file_name: str) -> pd.Series:
    if "file_name" in df.columns:
        m = df["file_name"].astype(str) == str(file_name)
        if m.any():
            return df.loc[m].iloc[0]

    if "BidsFolder" in df.columns and "SessionID" in df.columns:
        m = (df["BidsFolder"].astype(str) == str(patient_id)) & (df["SessionID"].astype(str) == str(session_id))
        if m.any():
            return df.loc[m].iloc[0]

    raise KeyError(f"Could not find demographics row for {file_name}")


def _build_single_record_batch(
    row: pd.Series,
    site_id: str,
    file_name: str,
    train_root: Path,
    basename_index: Dict[str, List[Path]],
    encoder: FixedDemographicsEncoder,
) -> Dict[str, torch.Tensor]:
    phys_file = train_root / "physiological_data" / site_id / file_name
    if not phys_file.exists():
        raise FileNotFoundError(f"Physiological EDF not found: {phys_file}")

    psg_data, psg_ch_names, psg_sfreq, _ = read_and_optionally_resample(
        phys_file, target_sfreq=TARGET_PSG_SFREQ, verbose=False
    )
    if abs(psg_sfreq - TARGET_PSG_SFREQ) > 1e-6:
        raise ValueError(f"Unexpected PSG sfreq after resample: {psg_sfreq}")

    grouped = group_channels_for_sleepfm(psg_ch_names, keep_unmatched=True, preserve_original_order=False)
    grouped_channels = grouped["grouped"]

    psg_arrays: Dict[str, np.ndarray] = {}
    psg_channel_names: Dict[str, List[str]] = {}
    psg_token_counts: Dict[str, int] = {}

    psg_data = psg_data.astype(np.float32, copy=False)
    for mod in MODALITIES:
        mod_key = mod.upper()
        channels = grouped_channels.get(mod_key, [])
        if len(channels) == 0:
            psg_arrays[mod] = np.zeros((0, 0, PSG_SAMPLES_PER_TOKEN), dtype=np.float32)
            psg_channel_names[mod] = []
            psg_token_counts[mod] = 0
            continue

        idxs = [int(ch.original_index) for ch in channels]
        names = [str(ch.raw_name) for ch in channels]
        arr = psg_data[idxs]
        n_tokens = int(arr.shape[1]) // PSG_SAMPLES_PER_TOKEN
        if n_tokens <= 0:
            psg_arrays[mod] = np.zeros((len(idxs), 0, PSG_SAMPLES_PER_TOKEN), dtype=np.float32)
            psg_channel_names[mod] = names
            psg_token_counts[mod] = 0
            continue

        arr = arr[:, : n_tokens * PSG_SAMPLES_PER_TOKEN]
        arr = arr.reshape(len(idxs), n_tokens, PSG_SAMPLES_PER_TOKEN).astype(np.float32, copy=False)
        psg_arrays[mod] = arr
        psg_channel_names[mod] = names
        psg_token_counts[mod] = n_tokens

    positive_psg_counts = [n for n in psg_token_counts.values() if n > 0]
    if len(positive_psg_counts) == 0:
        raise ValueError(f"No usable PSG tokens for {file_name}")

    caisr_path, _ = find_matching_annotation_files(phys_file, basename_index)
    has_caisr = False
    caisr_tok = np.zeros((0, len(CAISR_CHANNELS)), dtype=np.float32)
    caisr_channel_mask = np.zeros((len(CAISR_CHANNELS),), dtype=np.bool_)

    if caisr_path is not None and Path(caisr_path).exists():
        ann_data, ann_ch_names, ann_sfreq, _ = read_and_optionally_resample(
            Path(caisr_path), target_sfreq=TARGET_ANN_SFREQ, verbose=False
        )
        if abs(ann_sfreq - TARGET_ANN_SFREQ) <= 1e-6:
            ann_tok = ann_to_token_level(ann_data)
            ann_tok, ann_mask = select_annotation_channels(ann_tok, ann_ch_names, CAISR_CHANNELS)
            caisr_channel_mask = ann_mask.astype(bool, copy=False)
            if ann_tok.shape[0] > 0 and bool(caisr_channel_mask.any()):
                has_caisr = True
                caisr_tok = ann_tok.astype(np.float32, copy=False)

    usable_candidates = list(positive_psg_counts)
    if has_caisr:
        usable_candidates.append(int(caisr_tok.shape[0]))

    usable_n = int(min(usable_candidates)) if len(usable_candidates) > 0 else 0
    if usable_n <= 0:
        raise ValueError(f"No usable tokens after alignment for {file_name}")

    chunk_tokens = 60
    n_chunks = int(np.ceil(usable_n / chunk_tokens))
    token_mask = np.zeros((1, n_chunks, chunk_tokens), dtype=np.bool_)
    chunk_mask = np.zeros((1, n_chunks), dtype=np.bool_)

    for j in range(n_chunks):
        st = j * chunk_tokens
        ed = min((j + 1) * chunk_tokens, usable_n)
        if ed > st:
            token_mask[0, j, : ed - st] = True
            chunk_mask[0, j] = True

    batch: Dict[str, Any] = {
        "file_name": [file_name],
        "site": [site_id],
        "h5_path": [""],
        "n_chunks": torch.tensor([n_chunks], dtype=torch.int64),
        "token_mask": torch.from_numpy(token_mask),
        "chunk_mask": torch.from_numpy(chunk_mask),
        "caisr_chunks": torch.zeros((1, n_chunks, chunk_tokens, len(CAISR_CHANNELS)), dtype=torch.float32),
        "caisr_channel_mask": torch.from_numpy(caisr_channel_mask.reshape(1, -1)),
        "has_caisr": torch.tensor([[1.0 if has_caisr else 0.0]], dtype=torch.float32),
        "demo_x": torch.from_numpy(encoder.transform_row(row).reshape(1, -1).astype(np.float32)),
        "y": torch.zeros((1,), dtype=torch.float32),
        "has_label": torch.tensor([False], dtype=torch.bool),
    }

    for mod in MODALITIES:
        arr = psg_arrays[mod]
        c = int(arr.shape[0])
        chunks = np.zeros((1, n_chunks, chunk_tokens, c, PSG_SAMPLES_PER_TOKEN), dtype=np.float32)
        n_tokens_mod = int(arr.shape[1])
        if c > 0 and n_tokens_mod > 0:
            tok = arr[:, : min(n_tokens_mod, usable_n), :]
            tok = tok.transpose(1, 0, 2)
            for j in range(n_chunks):
                st = j * chunk_tokens
                ed = min((j + 1) * chunk_tokens, tok.shape[0], usable_n)
                if ed > st:
                    chunks[0, j, : ed - st] = tok[st:ed]

        batch[f"{mod}_chunks"] = torch.from_numpy(chunks)
        batch[f"{mod}_channel_mask"] = torch.ones((1, c), dtype=torch.bool)
        batch[f"{mod}_channel_names"] = [psg_channel_names[mod]]

    if has_caisr:
        for j in range(n_chunks):
            st = j * chunk_tokens
            ed = min((j + 1) * chunk_tokens, caisr_tok.shape[0], usable_n)
            if ed > st:
                batch["caisr_chunks"][0, j, : ed - st] = torch.from_numpy(caisr_tok[st:ed].astype(np.float32))

    return batch


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out
