#!/usr/bin/env python3
"""V14 with one joint record-wise SessionID/Age/BMI/Sex residual."""

from __future__ import annotations

import json
import hashlib
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
MODEL_SUBDIR = "raw_sequence_v15_joint_recordwise_demographics"
DEFAULT_THRESHOLD = 0.5
ADAPTATION_RECORDS = 6
ADAPTATION_LEARNING_RATE = 1e-6
DATE_RULE_FILENAME = "date_residual.json"
CAISR_RULE_FILENAME = "caisr_residual.json"
FOLLOWUP_RULE_FILENAME = "followup_residual.json"
DEMOGRAPHICS_RULE_FILENAME = "recordwise_demographics_residual.json"
AGE_GAP = 2.0
FOLLOW_UP_HORIZON_DAYS = 2192.0
DAYS_PER_YEAR = 365.25
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


def _followup_adjustment(rule: dict[str, Any], row: pd.Series | dict) -> float:
    value = _row_value(row, ["CreationTime", "creation_time"], False)
    risk = _followup_risk(rule, _creation_day(value))
    return float(rule["risk_coefficient"]) * risk


def _demographics_adjustment(rule: dict[str, Any], row: pd.Series | dict) -> float:
    """Apply the joint rule to observed current-record fields only."""
    adjustment = 0.0

    session = pd.to_numeric(
        _row_value(row, ["SessionID", "session_id", "session"], False),
        errors="coerce",
    )
    if session is not None and np.isfinite(session):
        session_rule = rule["session"]
        feature = math.log1p(max(float(session) - 1.0, 0.0))
        adjustment += float(session_rule["coefficient"]) * (
            (feature - float(session_rule["mean"])) / float(session_rule["std"])
        )

    age = pd.to_numeric(_row_value(row, ["Age", "age"], False), errors="coerce")
    if age is not None and np.isfinite(age):
        age_rule = rule["age"]
        adjustment += float(age_rule["coefficient"]) * (
            (float(age) - float(age_rule["mean"])) / float(age_rule["std"])
        )

    bmi = pd.to_numeric(_row_value(row, ["BMI", "bmi"], False), errors="coerce")
    if bmi is not None and np.isfinite(bmi):
        bmi_rule = rule["bmi"]
        adjustment += float(bmi_rule["coefficient"]) * (
            (float(bmi) - float(bmi_rule["mean"])) / float(bmi_rule["std"])
        )

    sex = _row_value(row, ["Sex", "sex"], False)
    sex_text = "" if sex is None else str(sex).strip().lower()
    if sex_text in {"male", "m"}:
        adjustment += float(rule["sex"]["male_coefficient"])

    return float(adjustment)


def _runtime_followup_adjustment(
    rule: dict[str, Any], row: pd.Series | dict, data_folder: Path
) -> float:
    value = _row_value(row, ["CreationTime", "creation_time"], False)
    if value is not None:
        return _followup_adjustment(rule, row)
    patient_id, site_id, session_id, _ = _record_parts(row)
    value = _date_lookup(data_folder).get((patient_id, site_id, session_id))
    return _followup_adjustment(rule, {"CreationTime": value})


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


def _copy_pretrained(model_folder: Path) -> Path:
    if not (PRETRAINED_DIR / "metadata.json").is_file():
        raise FileNotFoundError(f"Packaged pretrained model is incomplete: {PRETRAINED_DIR}")
    destination = model_folder / MODEL_SUBDIR
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PRETRAINED_DIR, destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refresh_model_metadata(
    model_root: Path,
    training_records: int,
    training_labels: list[int],
    date_rule: dict[str, Any],
    followup_rule: dict[str, Any],
    demographics_rule: dict[str, Any],
) -> None:
    metadata_path = model_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_records"] = int(training_records)
    metadata["training_labeled_records"] = len(training_labels)
    metadata["training_positives"] = int(sum(training_labels))
    metadata["creation_time_residual"]["fit_source"] = date_rule.get("source")
    metadata["followup_residual"]["fit_source"] = followup_rule.get("source")
    metadata["recordwise_demographics_residual"] = {
        "fit_source": demographics_rule.get("source"),
        "fields": ["SessionID", "Age", "BMI", "Sex"],
        "record_wise_inference": True,
        "per_field_missing_behavior": "zero adjustment",
        "explicit_missingness_features": False,
    }
    metadata.setdefault("files", {}).setdefault(DEMOGRAPHICS_RULE_FILENAME, {})
    for filename, file_metadata in metadata.get("files", {}).items():
        path = model_root / filename
        if path.is_file():
            file_metadata["bytes"] = path.stat().st_size
            file_metadata["sha256"] = _sha256(path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    try:
        followup_rule = _fit_followup_rule(frame)
    except ValueError as exception:
        followup_rule = json.loads(
            (PRETRAINED_DIR / FOLLOWUP_RULE_FILENAME).read_text(encoding="utf-8")
        )
        followup_rule["source"] = "packaged_fallback_for_incomplete_training_fields"
        followup_rule["fallback_reason"] = str(exception)
    (model_root / FOLLOWUP_RULE_FILENAME).write_text(
        json.dumps(followup_rule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    demographics_rule = json.loads(
        (PRETRAINED_DIR / DEMOGRAPHICS_RULE_FILENAME).read_text(encoding="utf-8")
    )
    (model_root / DEMOGRAPHICS_RULE_FILENAME).write_text(
        json.dumps(demographics_rule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    _refresh_model_metadata(
        model_root,
        training_records=len(frame),
        training_labels=[label for _, label in candidates],
        date_rule=date_rule,
        followup_rule=followup_rule,
        demographics_rule=demographics_rule,
    )
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
                "followup_residual": followup_rule,
                "recordwise_demographics_residual": demographics_rule,
                "residual_blend_weight": 0.5,
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
    runtime["followup_rule"] = json.loads(
        (model_root / FOLLOWUP_RULE_FILENAME).read_text(encoding="utf-8")
    )
    runtime["demographics_rule"] = json.loads(
        (model_root / DEMOGRAPHICS_RULE_FILENAME).read_text(encoding="utf-8")
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

    try:
        date_adjustment = _runtime_date_adjustment(
            model["date_rule"], record, data_folder
        )
    except Exception:
        date_adjustment = 0.0
    if psg_path is None:
        caisr_adjustment, caisr_status = 0.0, "raw-unavailable"
    else:
        caisr_adjustment, caisr_status = _caisr_adjustment(
            model["caisr_rule"], psg_path, caisr_path, record_id
        )
    try:
        followup_adjustment = _runtime_followup_adjustment(
            model["followup_rule"], record, data_folder
        )
    except Exception:
        followup_adjustment = 0.0
    logit += 0.5 * (date_adjustment + caisr_adjustment + followup_adjustment)
    try:
        demographics_adjustment = _demographics_adjustment(
            model["demographics_rule"], record
        )
    except Exception:
        demographics_adjustment = 0.0
    logit += demographics_adjustment
    if not np.isfinite(logit):
        logit = 0.0
    probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
    if verbose:
        print(
            f"{patient_id}: {diagnostics['eligible_epoch_count']}/"
            f"{diagnostics['complete_epoch_count']} eligible epochs; "
            f"Raw={raw_status}; "
            f"date adjustment={date_adjustment:.4f}; "
            f"CAISR adjustment={caisr_adjustment:.4f} ({caisr_status}); "
            f"follow-up adjustment={followup_adjustment:.4f}; "
            f"recordwise-demographics adjustment={demographics_adjustment:.4f}; "
            "V14 residual blend=0.5",
            flush=True,
        )
    return bool(probability >= DEFAULT_THRESHOLD), float(probability)
