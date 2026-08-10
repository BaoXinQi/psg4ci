#!/usr/bin/env python3
"""Domain-Raw full-night model with record-wise date and CAISR residuals."""

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

import online_features
from raw_sequence_runtime import load_runtime, predict_psg


SCRIPT_DIR = Path(__file__).resolve().parent
PRETRAINED_DIR = SCRIPT_DIR / "pretrained_raw"
MODEL_SUBDIR = "raw_sequence_v13_domain_date_caisr"
DEFAULT_THRESHOLD = 0.5
ADAPTATION_RECORDS = 6
ADAPTATION_LEARNING_RATE = 1e-6
DATE_RULE_FILENAME = "date_residual.json"
CAISR_RULE_FILENAME = "caisr_residual.json"
AGE_GAP = 2.0
_DATE_LOOKUP_CACHE: dict[str, dict[tuple[str, str, str], Any]] = {}


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


def _creation_day(value: Any) -> float:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return float("nan")
    return float(parsed.value / 86_400_000_000_000.0)


def _legal_date_differences(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=int)
    ages = frame["age"].to_numpy(dtype=float)
    dates = frame["date_z"].to_numpy(dtype=float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    legal = np.abs(ages[positive, None] - ages[negative][None, :]) <= AGE_GAP
    return (dates[positive, None] - dates[negative][None, :])[legal]


def _fit_positive_site_macro_coefficient(groups: list[np.ndarray]) -> float:
    if not groups or any(group.size == 0 for group in groups):
        raise ValueError("Each site must contain legal age-matched positive-negative pairs")
    coefficient = 1.0
    site_weight = 1.0 / len(groups)
    for _ in range(50):
        gradient = 0.0
        hessian = 0.0
        for differences in groups:
            scaled = np.clip(coefficient * differences, -60.0, 60.0)
            probability = 1.0 / (1.0 + np.exp(-scaled))
            gradient += site_weight * float(
                np.mean(-differences * (1.0 - probability))
            )
            hessian += site_weight * float(
                np.mean(differences**2 * probability * (1.0 - probability))
            )
        if hessian <= 1e-12:
            break
        updated = float(np.clip(coefficient - gradient / hessian, 0.0, 10.0))
        if abs(updated - coefficient) <= 1e-10:
            coefficient = updated
            break
        coefficient = updated
    return coefficient


def _fit_date_rule(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in frame.to_dict(orient="records"):
        label = _parse_binary(
            _row_value(
                row,
                ["Cognitive_Impairment", "cognitive_impairment", "label"],
                False,
            )
        )
        age_value = pd.to_numeric(
            _row_value(row, ["Age", "age"], False), errors="coerce"
        )
        age = float(age_value) if age_value is not None else float("nan")
        site = _clean_identifier(_row_value(row, ["SiteID", "site_id", "site"], False))
        day = _creation_day(
            _row_value(row, ["CreationTime", "creation_time"], False)
        )
        if label is not None and np.isfinite(age) and site and np.isfinite(day):
            rows.append({"label": label, "age": float(age), "site": site, "day": day})
    working = pd.DataFrame(rows)
    if working.empty:
        raise ValueError("No complete labeled rows are available for the date residual")
    mean_day = float(working["day"].mean())
    std_day = max(float(working["day"].to_numpy().std()), 1e-6)
    working["date_z"] = (working["day"] - mean_day) / std_day
    groups = []
    pair_counts = {}
    for site, site_frame in working.groupby("site", sort=True):
        differences = _legal_date_differences(site_frame.reset_index(drop=True))
        pair_counts[str(site)] = int(differences.size)
        if differences.size:
            groups.append(differences)
    if sum(pair_counts.values()) < 100:
        raise ValueError(f"Too few legal date pairs: {pair_counts}")
    coefficient = _fit_positive_site_macro_coefficient(groups)
    return {
        "version": "creation_time_site_macro_pairwise_v1",
        "source": "official_training_labels",
        "age_gap_years": AGE_GAP,
        "creation_day_mean": mean_day,
        "creation_day_std": std_day,
        "date_coefficient": coefficient,
        "training_pair_counts": pair_counts,
    }


def _date_adjustment(rule: dict[str, Any], row: pd.Series | dict) -> float:
    value = _row_value(row, ["CreationTime", "creation_time"], False)
    day = _creation_day(value)
    if not np.isfinite(day):
        return 0.0
    standardized = (
        day - float(rule["creation_day_mean"])
    ) / float(rule["creation_day_std"])
    return float(rule["date_coefficient"]) * standardized


def _date_lookup(data_folder: Path) -> dict[tuple[str, str, str], Any]:
    path = _demographics_path(data_folder).resolve()
    key = str(path)
    if key not in _DATE_LOOKUP_CACHE:
        frame = pd.read_csv(path)
        lookup = {}
        for row in frame.to_dict(orient="records"):
            patient_id = _clean_identifier(
                _row_value(
                    row,
                    ["BidsFolder", "bids_folder", "patient_id", "PatientID"],
                )
            )
            site_id = _clean_identifier(
                _row_value(row, ["SiteID", "site_id", "site"])
            )
            session_id = _clean_identifier(
                _row_value(row, ["SessionID", "session_id", "session"])
            )
            lookup[(patient_id, site_id, session_id)] = _row_value(
                row, ["CreationTime", "creation_time"], False
            )
        _DATE_LOOKUP_CACHE[key] = lookup
    return _DATE_LOOKUP_CACHE[key]


def _runtime_date_adjustment(
    rule: dict[str, Any], row: pd.Series | dict, data_folder: Path
) -> float:
    direct = _date_adjustment(rule, row)
    if direct != 0.0 or _row_value(
        row, ["CreationTime", "creation_time"], False
    ) is not None:
        return direct
    patient_id, site_id, session_id, _ = _record_parts(row)
    value = _date_lookup(data_folder).get((patient_id, site_id, session_id))
    return _date_adjustment(rule, {"CreationTime": value})


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


def _record_paths(
    data_root: Path, site_id: str, record_id: str
) -> tuple[Path, Path | None]:
    psg_path = data_root / "physiological_data" / site_id / f"{record_id}.edf"
    if not psg_path.is_file():
        matches = list((data_root / "physiological_data").glob(f"*/{record_id}.edf"))
        if len(matches) != 1:
            raise FileNotFoundError(f"PSG not found for {record_id}")
        psg_path = matches[0]
    caisr_path = (
        data_root
        / "algorithmic_annotations"
        / site_id
        / f"{record_id}_caisr_annotations.edf"
    )
    if not caisr_path.is_file():
        matches = list(
            (data_root / "algorithmic_annotations").glob(
                f"*/{record_id}_caisr_annotations.edf"
            )
        )
        caisr_path = matches[0] if len(matches) == 1 else None
    return psg_path, caisr_path


def _numeric_feature(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


def _caisr_adjustment(
    rule: dict[str, Any],
    psg_path: Path,
    caisr_path: Path | None,
    record_id: str,
) -> tuple[float, str]:
    if caisr_path is None:
        return 0.0, "missing"
    try:
        features = online_features.extract_caisr_features(
            psg_path, caisr_path, record_id
        )
    except Exception:
        return 0.0, "failed"
    if _numeric_feature(features.get("caisr_file_available", 0.0)) <= 0.0:
        return 0.0, "unavailable"
    transform = rule["feature_transform"]
    columns = [str(value) for value in rule["columns"]]
    values = np.asarray(
        [_numeric_feature(features.get(column)) for column in columns], dtype=float
    )
    median = np.asarray(transform["median"], dtype=float)
    mean = np.asarray(transform["mean"], dtype=float)
    std = np.asarray(transform["std"], dtype=float)
    values = np.where(np.isfinite(values), values, median)
    normalized = (values - mean) / std
    coefficients = np.asarray(rule["feature_coefficients"], dtype=float)
    return float(normalized @ coefficients), "ok"


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
    try:
        date_rule = _fit_date_rule(frame)
    except ValueError as exception:
        date_rule = json.loads(
            (PRETRAINED_DIR / DATE_RULE_FILENAME).read_text(encoding="utf-8")
        )
        date_rule["source"] = "packaged_fallback_for_insufficient_training_pairs"
        date_rule["fallback_reason"] = str(exception)
    (model_root / DATE_RULE_FILENAME).write_text(
        json.dumps(date_rule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
            path, _ = _record_paths(data_root, site_id, record_id)
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
                "date_residual": date_rule,
                "caisr_residual": "packaged_full_large_training_fit",
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
    runtime["date_rule"] = json.loads(
        (model_root / DATE_RULE_FILENAME).read_text(encoding="utf-8")
    )
    runtime["caisr_rule"] = json.loads(
        (model_root / CAISR_RULE_FILENAME).read_text(encoding="utf-8")
    )
    if verbose:
        print(
            f"Loaded frozen E1 + {len(runtime['sequences'])}-member Raw full-night ensemble",
            flush=True,
        )
    return runtime


def run_model(model, record, data_folder, verbose):
    patient_id, site_id, _, record_id = _record_parts(record)
    data_root = _data_root(Path(data_folder))
    psg_path, caisr_path = _record_paths(data_root, site_id, record_id)
    logit, diagnostics = predict_psg(model, psg_path, record_id, site_id)
    date_adjustment = _runtime_date_adjustment(
        model["date_rule"], record, Path(data_folder)
    )
    caisr_adjustment, caisr_status = _caisr_adjustment(
        model["caisr_rule"], psg_path, caisr_path, record_id
    )
    logit += date_adjustment + caisr_adjustment
    probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
    if verbose:
        print(
            f"{patient_id}: {diagnostics['eligible_epoch_count']}/"
            f"{diagnostics['complete_epoch_count']} eligible epochs; "
            f"date adjustment={date_adjustment:.4f}; "
            f"CAISR adjustment={caisr_adjustment:.4f} ({caisr_status})",
            flush=True,
        )
    return bool(probability >= DEFAULT_THRESHOLD), float(probability)
