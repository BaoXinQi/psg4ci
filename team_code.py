#!/usr/bin/env python3
"""V14 Domain-Raw model with a weaker record-wise residual."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyedflib

from helper_code import *  # noqa: F401,F403

import online_features
from full_training_constants import MODEL_SUBDIR, RESIDUAL_BLEND_WEIGHT
from raw_sequence_runtime import load_runtime, predict_psg


DEFAULT_THRESHOLD = 0.5
DATE_RULE_FILENAME = "date_residual.json"
CAISR_RULE_FILENAME = "caisr_residual.json"
FOLLOWUP_RULE_FILENAME = "followup_residual.json"
AGE_GAP = 2.0
FOLLOW_UP_HORIZON_DAYS = 2192.0
DAYS_PER_YEAR = 365.25


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


def _followup_risk(rule: dict[str, Any], day: float) -> float:
    if not np.isfinite(day):
        return 0.0
    cutoffs = np.asarray(list(rule["training_site_cutoffs"].values()), dtype=float)
    thresholds = cutoffs - float(rule["follow_up_horizon_days"])
    risks = np.maximum(day - thresholds, 0.0) / float(rule["days_per_year"])
    return float(np.mean(risks))


def _legal_followup_differences(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=int)
    ages = frame["age"].to_numpy(dtype=float)
    risks = frame["risk"].to_numpy(dtype=float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    legal = np.abs(ages[positive, None] - ages[negative][None, :]) <= AGE_GAP
    return (risks[positive, None] - risks[negative][None, :])[legal]


def _fit_followup_rule(frame: pd.DataFrame) -> dict[str, Any]:
    cutoff_rows = []
    labeled_rows = []
    for row in frame.to_dict(orient="records"):
        site = _clean_identifier(
            _row_value(row, ["SiteID", "site_id", "site"], False)
        )
        creation = _creation_day(
            _row_value(row, ["CreationTime", "creation_time"], False)
        )
        last_visit = _creation_day(
            _row_value(
                row,
                ["Last_Known_Visit_Date", "last_known_visit_date"],
                False,
            )
        )
        if site and np.isfinite(last_visit):
            cutoff_rows.append({"site": site, "last_visit": last_visit})

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
        if label is not None and site and np.isfinite(age) and np.isfinite(creation):
            labeled_rows.append(
                {"label": label, "age": age, "site": site, "day": creation}
            )

    cutoff_frame = pd.DataFrame(cutoff_rows)
    working = pd.DataFrame(labeled_rows)
    if cutoff_frame.empty or working.empty:
        raise ValueError("Follow-up rule requires labeled rows and last-visit dates")

    site_cutoffs = {
        str(site): float(np.quantile(site_frame["last_visit"], 0.99))
        for site, site_frame in cutoff_frame.groupby("site", sort=True)
    }
    if len(site_cutoffs) < 2:
        raise ValueError("Follow-up rule requires at least two training sites")
    provisional_rule = {
        "training_site_cutoffs": site_cutoffs,
        "follow_up_horizon_days": FOLLOW_UP_HORIZON_DAYS,
        "days_per_year": DAYS_PER_YEAR,
    }
    working["risk"] = working["day"].map(
        lambda value: _followup_risk(provisional_rule, float(value))
    )
    groups = []
    pair_counts = {}
    for site, site_frame in working.groupby("site", sort=True):
        differences = _legal_followup_differences(
            site_frame.reset_index(drop=True)
        )
        pair_counts[str(site)] = int(differences.size)
        if differences.size:
            groups.append(differences)
    if sum(pair_counts.values()) < 100:
        raise ValueError(f"Too few legal follow-up pairs: {pair_counts}")
    coefficient = _fit_positive_site_macro_coefficient(groups)
    return {
        "version": "recordwise_training_cutoff_marginalized_6y_v1",
        "source": "official_training_labels_and_last_visit_dates",
        "record_wise_inference": True,
        "cutoff_quantile": 0.99,
        "cutoff_aggregation": "equal_mean_of_site_specific_risks",
        "follow_up_horizon_days": FOLLOW_UP_HORIZON_DAYS,
        "days_per_year": DAYS_PER_YEAR,
        "training_site_cutoffs": site_cutoffs,
        "risk_coefficient": coefficient,
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


def _edf_creation_time(psg_path: Path | None) -> Any:
    """Read the current record's normalized EDF start time, or fail closed."""
    if psg_path is None or not Path(psg_path).is_file():
        return None
    try:
        with pyedflib.EdfReader(str(psg_path)) as reader:
            value = reader.getStartdatetime()
    except Exception:
        return None
    return value if np.isfinite(_creation_day(value)) else None


def _runtime_creation_time(
    row: pd.Series | dict, psg_path: Path | None
) -> tuple[Any, str]:
    value = _row_value(row, ["CreationTime", "creation_time"], False)
    if np.isfinite(_creation_day(value)):
        return value, "metadata"
    value = _edf_creation_time(psg_path)
    if np.isfinite(_creation_day(value)):
        return value, "edf-header"
    return None, "missing"


def _followup_adjustment(rule: dict[str, Any], row: pd.Series | dict) -> float:
    value = _row_value(row, ["CreationTime", "creation_time"], False)
    risk = _followup_risk(rule, _creation_day(value))
    return float(rule["risk_coefficient"]) * risk


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
    support_columns = (
        "caisr_stage_valid_epoch_fraction",
        "caisr_stage_probability_valid_epoch_fraction",
        "caisr_arousal_valid_epoch_fraction",
        "caisr_respiratory_valid_epoch_fraction",
        "caisr_limb_valid_epoch_fraction",
    )
    support = [_numeric_feature(features.get(column)) for column in support_columns]
    if not any(np.isfinite(value) and value > 0.0 for value in support):
        return 0.0, "invalid"
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


def train_model(data_folder, model_folder, verbose):
    """Reproduce the complete Challenge-data training pipeline on the server."""
    from full_training_pipeline import run_full_training

    run_full_training(Path(data_folder), Path(model_folder), bool(verbose))


def load_model(model_folder, verbose):
    model_root = Path(model_folder) / MODEL_SUBDIR
    runtime = load_runtime(model_root)
    runtime["date_rule"] = json.loads(
        (model_root / DATE_RULE_FILENAME).read_text(encoding="utf-8")
    )
    runtime["caisr_rule"] = json.loads(
        (model_root / CAISR_RULE_FILENAME).read_text(encoding="utf-8")
    )
    runtime["followup_rule"] = json.loads(
        (model_root / FOLLOWUP_RULE_FILENAME).read_text(encoding="utf-8")
    )
    if verbose:
        print(
            f"Loaded frozen E1 + {len(runtime['sequences'])}-member Raw full-night ensemble",
            flush=True,
        )
    return runtime


def run_model(model, record, data_folder, verbose):
    data_folder = Path(data_folder)
    patient_id = _clean_identifier(
        _row_value(
            record,
            ["BidsFolder", "bids_folder", "patient_id", "PatientID"],
            False,
        )
    )
    record_id = patient_id or "unknown"
    psg_path = None
    caisr_path = None
    diagnostics = {"eligible_epoch_count": 0, "complete_epoch_count": 0}
    raw_status = "fallback"
    try:
        patient_id, site_id, _, record_id = _record_parts(record)
        data_root = _data_root(data_folder)
        psg_path, caisr_path = _record_paths(data_root, site_id, record_id)
        logit, diagnostics = predict_psg(model, psg_path, record_id, site_id)
        if not np.isfinite(logit):
            raise FloatingPointError("Non-finite Raw CI logit")
        raw_status = "ok"
    except Exception as exception:
        logit = 0.0
        raw_status = f"fallback:{type(exception).__name__}"

    creation_time, creation_time_source = _runtime_creation_time(record, psg_path)
    resolved_creation = {"CreationTime": creation_time}
    try:
        date_adjustment = _date_adjustment(model["date_rule"], resolved_creation)
    except Exception:
        date_adjustment = 0.0
    if psg_path is None:
        caisr_adjustment, caisr_status = 0.0, "raw-unavailable"
    else:
        caisr_adjustment, caisr_status = _caisr_adjustment(
            model["caisr_rule"], psg_path, caisr_path, record_id
        )
    try:
        followup_adjustment = _followup_adjustment(
            model["followup_rule"], resolved_creation
        )
    except Exception:
        followup_adjustment = 0.0
    logit += RESIDUAL_BLEND_WEIGHT * (
        date_adjustment + caisr_adjustment + followup_adjustment
    )
    if not np.isfinite(logit):
        logit = 0.0
    probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
    if verbose:
        print(
            f"{patient_id}: {diagnostics['eligible_epoch_count']}/"
            f"{diagnostics['complete_epoch_count']} eligible epochs; "
            f"Raw={raw_status}; "
            f"CreationTime={creation_time_source}; "
            f"date adjustment={date_adjustment:.4f}; "
            f"CAISR adjustment={caisr_adjustment:.4f} ({caisr_status}); "
            f"follow-up adjustment={followup_adjustment:.4f}; "
            f"residual blend={RESIDUAL_BLEND_WEIGHT}",
            flush=True,
        )
    return bool(probability >= DEFAULT_THRESHOLD), float(probability)
