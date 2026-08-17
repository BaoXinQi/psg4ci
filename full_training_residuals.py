"""Refit every V17 record-wise residual from the provided training set."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from full_training_constants import CAISR_FEATURE_COLUMNS, CAISR_L2


AGE_GAP = 2.0
FOLLOW_UP_HORIZON_DAYS = 2192.0
DAYS_PER_YEAR = 365.25


def creation_day(values: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    output = parsed.astype("int64", copy=False).to_numpy(dtype=np.float64)
    output[parsed.isna().to_numpy()] = np.nan
    return output / 86_400_000_000_000.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def pair_differences(
    frame: pd.DataFrame, values: np.ndarray
) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for site, positions in frame.groupby("SiteID", sort=True).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        fold = frame.iloc[positions]
        labels = fold["label"].to_numpy(dtype=int)
        ages = fold["Age"].to_numpy(dtype=float)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        legal = np.abs(ages[positive, None] - ages[negative][None, :]) <= AGE_GAP
        positive_row, negative_row = np.nonzero(legal)
        local = values[positions]
        groups[str(site)] = (
            local[positive[positive_row]] - local[negative[negative_row]]
        )
    if not groups or any(len(value) == 0 for value in groups.values()):
        raise ValueError("Each site must contain legal age-matched positive-negative pairs")
    return groups


def fit_positive_coefficient(groups: dict[str, np.ndarray]) -> float:
    coefficient = 1.0
    for _ in range(50):
        gradient = 0.0
        hessian = 0.0
        for differences in groups.values():
            values = differences[:, 0]
            probability = sigmoid(coefficient * values)
            gradient += float(np.mean(-values * (1.0 - probability))) / len(groups)
            hessian += float(
                np.mean(values**2 * probability * (1.0 - probability))
            ) / len(groups)
        if hessian <= 1e-12:
            break
        updated = float(np.clip(coefficient - gradient / hessian, 0.0, 10.0))
        if abs(updated - coefficient) <= 1e-10:
            coefficient = updated
            break
        coefficient = updated
    return coefficient


def residual_objective(
    coefficients: np.ndarray,
    date_coefficient: float,
    date_groups: dict[str, np.ndarray],
    feature_groups: dict[str, np.ndarray],
    l2: float,
) -> float:
    loss = 0.0
    for site in date_groups:
        margin = date_coefficient * date_groups[site][:, 0] + feature_groups[site] @ coefficients
        loss += float(np.mean(np.logaddexp(0.0, -margin))) / len(date_groups)
    return loss + 0.5 * l2 * float(coefficients @ coefficients)


def fit_residual_coefficients(
    date_coefficient: float,
    date_groups: dict[str, np.ndarray],
    feature_groups: dict[str, np.ndarray],
    dimension: int,
    l2: float,
) -> np.ndarray:
    coefficients = np.zeros(dimension, dtype=np.float64)
    identity = np.eye(dimension, dtype=np.float64)
    for _ in range(60):
        gradient = l2 * coefficients
        hessian = l2 * identity
        for site in date_groups:
            values = feature_groups[site]
            margin = date_coefficient * date_groups[site][:, 0] + values @ coefficients
            probability = sigmoid(margin)
            weight = probability * (1.0 - probability)
            scale = 1.0 / (len(date_groups) * len(values))
            gradient += scale * (values.T @ (probability - 1.0))
            hessian += scale * ((values.T * weight) @ values)
        step = np.linalg.solve(hessian + 1e-8 * identity, gradient)
        current = residual_objective(
            coefficients, date_coefficient, date_groups, feature_groups, l2
        )
        step_scale = 1.0
        while step_scale > 1e-5:
            candidate = coefficients - step_scale * step
            if residual_objective(
                candidate, date_coefficient, date_groups, feature_groups, l2
            ) <= current:
                break
            step_scale *= 0.5
        updated = coefficients - step_scale * step
        if np.linalg.norm(updated - coefficients) < 1e-8:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


class FeatureTransform:
    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns
        self.median = np.zeros(len(columns), dtype=float)
        self.mean = np.zeros(len(columns), dtype=float)
        self.std = np.ones(len(columns), dtype=float)

    def fit(self, frame: pd.DataFrame) -> "FeatureTransform":
        values = frame[list(self.columns)].to_numpy(dtype=float)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(values, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(values), values, self.median)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std = np.where(self.std > 1e-6, self.std, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.median)
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def complete_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output["label"] = pd.to_numeric(output["label"], errors="raise").astype(int)
    output["Age"] = pd.to_numeric(output["Age"], errors="coerce")
    output["SiteID"] = output["SiteID"].astype(str)
    output["_creation_day"] = creation_day(output["CreationTime"])
    return output


def fit_date_rule(frame: pd.DataFrame) -> dict[str, Any]:
    working = frame.loc[np.isfinite(frame["Age"]) & np.isfinite(frame["_creation_day"])].copy()
    if working.empty:
        raise ValueError("No complete rows are available for the date residual")
    mean_day = float(working["_creation_day"].mean())
    std_day = max(float(working["_creation_day"].to_numpy().std()), 1e-6)
    date_z = ((working["_creation_day"].to_numpy() - mean_day) / std_day)[:, None]
    groups = pair_differences(working.reset_index(drop=True), date_z)
    return {
        "version": "creation_time_site_macro_pairwise_v1",
        "source": "official_training_labels",
        "age_gap_years": AGE_GAP,
        "creation_day_mean": mean_day,
        "creation_day_std": std_day,
        "date_coefficient": fit_positive_coefficient(groups),
        "training_pair_counts": {site: int(len(values)) for site, values in groups.items()},
    }


def fit_caisr_rule(frame: pd.DataFrame) -> dict[str, Any]:
    working = frame.loc[np.isfinite(frame["Age"]) & np.isfinite(frame["_creation_day"])].copy()
    day = working["_creation_day"].to_numpy(dtype=float)
    day_mean = float(day.mean())
    day_std = max(float(day.std()), 1e-6)
    date_values = ((day - day_mean) / day_std)[:, None]
    date_groups = pair_differences(working.reset_index(drop=True), date_values)
    date_coefficient = fit_positive_coefficient(date_groups)
    transform = FeatureTransform(CAISR_FEATURE_COLUMNS).fit(working)
    feature_values = transform.transform(working)
    feature_groups = pair_differences(working.reset_index(drop=True), feature_values)
    coefficients = fit_residual_coefficients(
        date_coefficient,
        date_groups,
        feature_groups,
        len(CAISR_FEATURE_COLUMNS),
        CAISR_L2,
    )
    return {
        "version": "caisr_temporal_date_conditional_v1",
        "source": "official_training_labels_and_caisr_annotations",
        "columns": list(CAISR_FEATURE_COLUMNS),
        "feature_transform": transform.to_dict(),
        "feature_coefficients": coefficients.tolist(),
        "conditional_date_coefficient": date_coefficient,
        "conditional_creation_day_mean": day_mean,
        "conditional_creation_day_std": day_std,
        "l2": CAISR_L2,
        "training_pair_counts": {site: int(len(values)) for site, values in date_groups.items()},
        "record_wise": True,
        "missing_behavior": "zero adjustment",
    }


def followup_risk(rule: dict[str, Any], day: float) -> float:
    if not np.isfinite(day):
        return 0.0
    cutoffs = np.asarray(list(rule["training_site_cutoffs"].values()), dtype=float)
    thresholds = cutoffs - float(rule["follow_up_horizon_days"])
    return float(np.mean(np.maximum(day - thresholds, 0.0) / float(rule["days_per_year"])))


def fit_followup_rule(frame: pd.DataFrame) -> dict[str, Any]:
    last_visit = creation_day(frame["Last_Known_Visit_Date"])
    cutoff_frame = frame.loc[np.isfinite(last_visit), ["SiteID"]].copy()
    cutoff_frame["last_visit"] = last_visit[np.isfinite(last_visit)]
    if cutoff_frame.empty:
        raise ValueError("No Last_Known_Visit_Date values are available")
    site_cutoffs = {
        str(site): float(np.quantile(group["last_visit"], 0.99))
        for site, group in cutoff_frame.groupby("SiteID", sort=True)
    }
    if len(site_cutoffs) < 2:
        raise ValueError("Follow-up fitting requires at least two sites")
    provisional = {
        "training_site_cutoffs": site_cutoffs,
        "follow_up_horizon_days": FOLLOW_UP_HORIZON_DAYS,
        "days_per_year": DAYS_PER_YEAR,
    }
    working = frame.loc[np.isfinite(frame["Age"]) & np.isfinite(frame["_creation_day"])].copy()
    risks = np.asarray(
        [followup_risk(provisional, day) for day in working["_creation_day"]], dtype=float
    )[:, None]
    groups = pair_differences(working.reset_index(drop=True), risks)
    return {
        "version": "recordwise_training_cutoff_marginalized_6y_v1",
        "source": "official_training_labels_and_last_visit_dates",
        "record_wise_inference": True,
        "cutoff_quantile": 0.99,
        "cutoff_aggregation": "equal_mean_of_site_specific_risks",
        "follow_up_horizon_days": FOLLOW_UP_HORIZON_DAYS,
        "days_per_year": DAYS_PER_YEAR,
        "training_site_cutoffs": site_cutoffs,
        "risk_coefficient": fit_positive_coefficient(groups),
        "training_pair_counts": {site: int(len(values)) for site, values in groups.items()},
    }


def fit_and_write_residuals(frame: pd.DataFrame, model_root: Path) -> dict[str, Any]:
    required = {"label", "Age", "SiteID", "CreationTime", "Last_Known_Visit_Date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Residual fitting is missing fields: {missing}")
    working = complete_labeled_frame(frame)
    date_rule = fit_date_rule(working)
    caisr_rule = fit_caisr_rule(working)
    followup_rule = fit_followup_rule(working)
    rules = {
        "date_residual.json": date_rule,
        "caisr_residual.json": caisr_rule,
        "followup_residual.json": followup_rule,
    }
    for filename, rule in rules.items():
        (model_root / filename).write_text(
            json.dumps(rule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {
        "date": date_rule,
        "caisr": {
            "features": len(CAISR_FEATURE_COLUMNS),
            "l2": CAISR_L2,
            "pair_counts": caisr_rule["training_pair_counts"],
        },
        "followup": followup_rule,
    }
