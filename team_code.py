#!/usr/bin/env python3
"""PhysioNet Challenge 2026 domain-robust Raw full-night entry."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from helper_code import *  # noqa: F401,F403

from raw_sequence_runtime import load_runtime, predict_psg


SCRIPT_DIR = Path(__file__).resolve().parent
PRETRAINED_DIR = SCRIPT_DIR / "pretrained_raw"
MODEL_SUBDIR = "raw_sequence_v4"
DEFAULT_THRESHOLD = 0.5
FALLBACK_PROBABILITY = 0.5
ADAPTATION_RECORDS = 6
ADAPTATION_LEARNING_RATE = 1e-6


def _clean_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _row_value(row: pd.Series | dict, names: list[str], required: bool = True) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip() != "":
            return row[name]
    if required:
        raise KeyError(f"Missing required field; tried {names}")
    return None


def _parse_binary(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0", "positive"}:
        return 1
    if text in {"false", "f", "no", "n", "0", "0.0", "negative"}:
        return 0
    return None


def _record_parts(row: pd.Series | dict) -> tuple[str, str, str, str]:
    patient_id = _clean_identifier(
        _row_value(row, ["BidsFolder", "bids_folder", "patient_id", "PatientID"])
    )
    site_id = _clean_identifier(_row_value(row, ["SiteID", "site_id", "site"]))
    session_id = _clean_identifier(_row_value(row, ["SessionID", "session_id", "session"]))
    return patient_id, site_id, session_id, f"{patient_id}_ses-{session_id}"


def _demographics_path(data_folder: Path) -> Path:
    for path in (data_folder / "demographics.csv", data_folder / "training_set" / "demographics.csv"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find demographics.csv under {data_folder}")


def _data_root(data_folder: Path) -> Path:
    if (data_folder / "physiological_data").is_dir():
        return data_folder
    if (data_folder / "training_set" / "physiological_data").is_dir():
        return data_folder / "training_set"
    raise FileNotFoundError(f"Could not find physiological_data under {data_folder}")


def _record_path(data_root: Path, site_id: str, record_id: str) -> Path:
    path = data_root / "physiological_data" / site_id / f"{record_id}.edf"
    if path.is_file():
        return path
    matches = list((data_root / "physiological_data").glob(f"*/{record_id}.edf"))
    if len(matches) != 1:
        raise FileNotFoundError(f"PSG not found for {record_id}")
    return matches[0]


def _copy_pretrained(model_folder: Path) -> Path:
    if not (PRETRAINED_DIR / "metadata.json").is_file():
        raise FileNotFoundError(f"Packaged pretrained model is incomplete: {PRETRAINED_DIR}")
    destination = model_folder / MODEL_SUBDIR
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PRETRAINED_DIR, destination)
    return destination


def train_model(data_folder, model_folder, verbose):
    data_folder = Path(data_folder)
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)
    model_root = _copy_pretrained(model_folder)
    frame = pd.read_csv(_demographics_path(data_folder))
    data_root = _data_root(data_folder)
    candidates: list[tuple[dict[str, Any], int]] = []
    for row in frame.to_dict(orient="records"):
        label = _parse_binary(
            _row_value(row, ["Cognitive_Impairment", "cognitive_impairment", "label"], False)
        )
        if label is not None:
            candidates.append((row, label))
    if not candidates:
        raise ValueError("No labeled training records were found")

    runtime = load_runtime(model_root)
    logits: list[float] = []
    labels: list[int] = []
    failures: list[dict[str, str]] = []
    for row, label in candidates:
        if len(logits) >= ADAPTATION_RECORDS:
            break
        try:
            _, site_id, _, record_id = _record_parts(row)
            path = _record_path(data_root, site_id, record_id)
            logit, _ = predict_psg(runtime, path, record_id, site_id)
            logits.append(logit)
            labels.append(label)
            if verbose:
                print(f"Raw adaptation audit: {len(logits)}/{ADAPTATION_RECORDS}", flush=True)
        except Exception as exception:
            failures.append({"record_id": str(row), "error": repr(exception)})
    if not logits:
        raise RuntimeError("No Raw adaptation audit record succeeded")

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits), -40.0, 40.0)))
    gradient = float(np.mean(probabilities - np.asarray(labels, dtype=float)))
    final_biases = [sequence.head[-1].bias for sequence in runtime["sequences"]]
    before = [float(bias.detach().item()) for bias in final_biases]
    with torch.no_grad():
        for bias in final_biases:
            bias.sub_(ADAPTATION_LEARNING_RATE * gradient)
    after = [float(bias.detach().item()) for bias in final_biases]
    for checkpoint_path, sequence, bias_before, bias_after in zip(
        runtime["sequence_paths"], runtime["sequences"], before, after
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["model_state"] = {
            name: value.detach().cpu() for name, value in sequence.state_dict().items()
        }
        checkpoint["official_training_adaptation"] = {
            "records": len(logits),
            "learning_rate": ADAPTATION_LEARNING_RATE,
            "bias_before": bias_before,
            "bias_after": bias_after,
        }
        torch.save(checkpoint, checkpoint_path)
    (model_folder / "training_metadata.json").write_text(
        json.dumps(
            {
                "labeled_records": len(candidates),
                "raw_audit_records": len(logits),
                "failed_audit_records": len(failures),
                "encoder_frozen": True,
                "sequence_ensemble_members": len(runtime["sequences"]),
                "sequences_frozen_except_final_bias": True,
                "adaptation_learning_rate": ADAPTATION_LEARNING_RATE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(failures).to_csv(model_folder / "training_failures.csv", index=False)


def load_model(model_folder, verbose):
    model_root = Path(model_folder) / MODEL_SUBDIR
    runtime = load_runtime(model_root)
    if verbose:
        print(
            f"Loaded frozen E1 + {len(runtime['sequences'])}-member Raw full-night ensemble",
            flush=True,
        )
    return runtime


def run_model(model, record, data_folder, verbose):
    patient_id = _clean_identifier(
        _row_value(record, ["BidsFolder", "bids_folder", "patient_id", "PatientID"], False)
    )
    try:
        patient_id, site_id, _, record_id = _record_parts(record)
        data_root = _data_root(Path(data_folder))
        psg_path = _record_path(data_root, site_id, record_id)
        logit, diagnostics = predict_psg(model, psg_path, record_id, site_id)
        probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
        if not np.isfinite(probability):
            raise FloatingPointError("Non-finite Raw CI probability")
        if verbose:
            print(
                f"{patient_id}: {diagnostics['eligible_epoch_count']}/"
                f"{diagnostics['complete_epoch_count']} eligible epochs",
                flush=True,
            )
    except Exception as exception:
        probability = FALLBACK_PROBABILITY
        if verbose:
            print(
                f"{patient_id or '<unknown>'}: Raw inference failed; "
                f"using neutral fallback ({exception!r})",
                flush=True,
            )
    return bool(probability >= DEFAULT_THRESHOLD), float(probability)
