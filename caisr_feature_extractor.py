#!/usr/bin/env python3
"""
Five-fold CAISR baselines for the PhysioNet Challenge 2026.

This script extracts whole-night features from the CAISR annotations stored in
the frozen full_v1 HDF5 files and evaluates two models with the frozen folds:

1. CAISR-only
2. Demographics + CAISR

For every fold, imputation, scaling, categorical encoding, and model fitting are
performed using only the training portion. Each eligible record receives one
out-of-fold prediction.

Default input:
    ~/fast_data/physionet2026/official_small/manifests/
    full_v1_split_5fold_v1.parquet

Default output:
    ~/fast_data/physionet2026/official_small/models/
    baseline_caisr_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VERSION = "baseline_caisr_v1"
AGE_GAP_YEARS = 2.0
SECONDS_PER_EPOCH = 30.0

STAGE_CODES = {
    "n3": 1,
    "n2": 2,
    "n1": 3,
    "rem": 4,
    "wake": 5,
}

STAGE_PROBABILITY_ALIASES = {
    "n3": [
        "prob_n3",
        "stage_prob_n3",
        "stage_probability_n3",
        "caisr_prob_n3",
    ],
    "n2": [
        "prob_n2",
        "stage_prob_n2",
        "stage_probability_n2",
        "caisr_prob_n2",
    ],
    "n1": [
        "prob_n1",
        "stage_prob_n1",
        "stage_probability_n1",
        "caisr_prob_n1",
    ],
    "rem": [
        "prob_r",
        "prob_rem",
        "stage_prob_r",
        "stage_prob_rem",
        "stage_probability_r",
        "stage_probability_rem",
        "caisr_prob_r",
        "caisr_prob_rem",
    ],
    "wake": [
        "prob_w",
        "prob_wake",
        "stage_prob_w",
        "stage_prob_wake",
        "stage_probability_w",
        "stage_probability_wake",
        "caisr_prob_w",
        "caisr_prob_wake",
    ],
}

LABEL_CANDIDATES = [
    "demographic__Cognitive_Impairment",
    "Cognitive_Impairment",
    "label",
]

AGE_CANDIDATES = [
    "demographic__Age",
    "Age",
]

SITE_CANDIDATES = [
    "demographic__SiteID",
    "SiteID",
    "site",
]

PATIENT_CANDIDATES = [
    "demographic__BDSPPatientID",
    "BDSPPatientID",
    "subject_id",
    "record_id",
]

RECORD_ID_CANDIDATES = [
    "record_id",
    "BidsFolder",
    "demographic__BidsFolder",
    "subject_id",
]

FOLD_CANDIDATES = [
    "fold",
    "cv_fold",
    "fold_id",
    "split_fold",
]

CACHE_PATH_CANDIDATES = [
    "cache_path",
    "cache_path_build",
    "full_v1_cache_path",
]

DEMOGRAPHIC_NUMERIC_CANDIDATES = {
    "Age": [
        "demographic__Age",
        "Age",
    ],
    "BMI": [
        "demographic__BMI",
        "BMI",
    ],
}

DEMOGRAPHIC_CATEGORICAL_CANDIDATES = {
    "Sex": [
        "demographic__Sex",
        "Sex",
    ],
    "Race": [
        "demographic__Race",
        "Race",
    ],
    "Ethnicity": [
        "demographic__Ethnicity",
        "Ethnicity",
    ],
}


def parse_arguments() -> argparse.Namespace:
    project_root = Path.home() / "fast_data/physionet2026"
    default_manifest = (
        project_root
        / "official_small/manifests/full_v1_split_5fold_v1.parquet"
    )
    default_cache_root = (
        project_root
        / "official_small/cache/full_v1/records"
    )
    default_output = (
        project_root
        / "official_small/models/baseline_caisr_v1"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Extract whole-night CAISR features and run frozen five-fold "
            "CAISR-only and demographics-plus-CAISR OOF baselines."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Frozen five-fold CSV or Parquet manifest.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=default_cache_root,
        help=(
            "Fallback directory containing <record_id>.h5 files when the "
            "manifest cache path is absent or stale."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for HDF5 feature extraction. Default: 4.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print feature-extraction progress every N records.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary predictions.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Logistic-regression class weighting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20262259,
        help="Random seed stored with the run.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum logistic-regression iterations.",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Re-extract CAISR features even if the feature file exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing model outputs.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported table format {suffix!r}; use CSV or Parquet."
        )

    if frame.empty:
        raise ValueError(f"Table is empty: {path}")

    return frame


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise ValueError(
            f"Could not find {description}. Tried: {list(candidates)}"
        )

    return None


def find_fold_column(frame: pd.DataFrame) -> str:
    column = find_column(
        frame,
        FOLD_CANDIDATES,
        "the fold column",
        required=False,
    )
    if column is not None:
        return column

    fuzzy = [
        str(value)
        for value in frame.columns
        if "fold" in str(value).lower()
        and "summary" not in str(value).lower()
    ]

    if len(fuzzy) == 1:
        return fuzzy[0]

    raise ValueError(
        "Could not uniquely identify the fold column. "
        f"Fold-like columns: {fuzzy}"
    )


def parse_binary_label(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))

    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return float(int(value))

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return float(value)

    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0", "positive"}:
        return 1.0
    if text in {"false", "f", "no", "n", "0", "0.0", "negative"}:
        return 0.0

    raise ValueError(f"Unrecognized binary label: {value!r}")


def clean_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))

    return str(value).strip()


def normalize_name(value: str) -> str:
    output = str(value).strip().lower()

    for character in [
        "/",
        "\\",
        "-",
        " ",
        ".",
        ":",
        "(",
        ")",
        "[",
        "]",
    ]:
        output = output.replace(character, "_")

    while "__" in output:
        output = output.replace("__", "_")

    return output.strip("_")


def as_1d_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)

    if array.ndim == 0:
        array = array.reshape(1)

    array = np.squeeze(array)

    if array.ndim != 1:
        return np.asarray([], dtype=float)

    try:
        return array.astype(float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def safe_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")

    return output if np.isfinite(output) else float("nan")


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=0)) if finite.size else float("nan")


def finite_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return (
        float(np.quantile(finite, quantile))
        if finite.size
        else float("nan")
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")

    return float(numerator / denominator)


def normalized_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]

    if counts.size == 0:
        return float("nan")

    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log(probabilities))

    if len(STAGE_CODES) <= 1:
        return float("nan")

    return float(entropy / np.log(len(STAGE_CODES)))


def run_count(values: np.ndarray, target: bool | None = None) -> int:
    values = np.asarray(values)

    if values.size == 0:
        return 0

    if target is not None:
        values = values.astype(bool)
        values = values == bool(target)

    changes = np.empty(values.size, dtype=bool)
    changes[0] = True
    changes[1:] = values[1:] != values[:-1]

    if target is None:
        return int(np.sum(changes))

    starts = changes & values.astype(bool)
    return int(np.sum(starts))


def longest_true_run(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool)

    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)


def collect_datasets(group: h5py.Group) -> dict[str, np.ndarray]:
    datasets: dict[str, np.ndarray] = {}

    def visitor(name: str, object_: Any) -> None:
        if isinstance(object_, h5py.Dataset):
            datasets[normalize_name(name)] = np.asarray(object_)

    group.visititems(visitor)
    return datasets


def find_dataset(
    datasets: Mapping[str, np.ndarray],
    aliases: Iterable[str],
) -> np.ndarray | None:
    normalized_aliases = [
        normalize_name(alias)
        for alias in aliases
    ]

    for alias in normalized_aliases:
        if alias in datasets:
            return datasets[alias]

    for alias in normalized_aliases:
        matches = [
            key
            for key in datasets
            if key.endswith("_" + alias)
        ]
        if len(matches) == 1:
            return datasets[matches[0]]

    return None


def read_boolean_series(
    datasets: Mapping[str, np.ndarray],
    aliases: Iterable[str],
    length: int,
) -> np.ndarray | None:
    value = find_dataset(datasets, aliases)

    if value is None:
        return None

    series = as_1d_numeric(value)

    if series.size != length:
        return None

    return np.isfinite(series) & (series > 0.5)


def read_numeric_series(
    datasets: Mapping[str, np.ndarray],
    aliases: Iterable[str],
    length: int,
) -> np.ndarray | None:
    value = find_dataset(datasets, aliases)

    if value is None:
        return None

    series = as_1d_numeric(value)

    if series.size != length:
        return None

    return series.astype(float)


def extract_stage_probability_matrix(
    datasets: Mapping[str, np.ndarray],
    n_epochs: int,
) -> tuple[np.ndarray | None, dict[str, str]]:
    columns: list[np.ndarray] = []
    sources: dict[str, str] = {}

    for stage_name in STAGE_CODES:
        selected_key: str | None = None
        selected_values: np.ndarray | None = None

        aliases = [
            normalize_name(alias)
            for alias in STAGE_PROBABILITY_ALIASES[stage_name]
        ]

        for alias in aliases:
            if alias in datasets:
                candidate = as_1d_numeric(
                    datasets[alias]
                )
                if candidate.size == n_epochs:
                    selected_key = alias
                    selected_values = candidate
                    break

        if selected_values is None:
            for alias in aliases:
                matching_keys = [
                    key
                    for key in datasets
                    if key.endswith("_" + alias)
                ]
                for key in matching_keys:
                    candidate = as_1d_numeric(
                        datasets[key]
                    )
                    if candidate.size == n_epochs:
                        selected_key = key
                        selected_values = candidate
                        break
                if selected_values is not None:
                    break

        if selected_values is None:
            columns = []
            break

        columns.append(
            selected_values.astype(float)
        )
        sources[stage_name] = str(
            selected_key
        )

    if len(columns) == len(STAGE_CODES):
        matrix = np.column_stack(columns)
        return matrix, sources

    matrix_aliases = {
        "stage_probabilities",
        "stage_probability",
        "stage_probs",
        "sleep_stage_probabilities",
        "sleep_stage_probability",
    }

    for key, value in datasets.items():
        if normalize_name(key) not in matrix_aliases:
            continue

        array = np.asarray(value, dtype=float)
        array = np.squeeze(array)

        if array.ndim != 2:
            continue

        if array.shape == (n_epochs, 5):
            return (
                array,
                {
                    stage_name: key
                    for stage_name in STAGE_CODES
                },
            )

        if array.shape == (5, n_epochs):
            return (
                array.T,
                {
                    stage_name: key
                    for stage_name in STAGE_CODES
                },
            )

    return None, {}


def extract_event_features(
    datasets: Mapping[str, np.ndarray],
    event_name: str,
    n_epochs: int,
    duration_hours: float,
) -> dict[str, float]:
    features: dict[str, float] = {}

    any_series = read_boolean_series(
        datasets,
        [
            f"{event_name}_any",
            f"{event_name}_event_any",
            f"{event_name}_positive",
        ],
        n_epochs,
    )

    valid_ratio = read_numeric_series(
        datasets,
        [
            f"{event_name}_valid_ratio",
            f"{event_name}_availability_ratio",
        ],
        n_epochs,
    )

    positive_ratio = read_numeric_series(
        datasets,
        [
            f"{event_name}_positive_ratio",
            f"{event_name}_event_ratio",
            f"{event_name}_fraction",
            f"{event_name}_mean",
            f"{event_name}_burden",
        ],
        n_epochs,
    )

    if valid_ratio is not None:
        valid_ratio = np.clip(
            valid_ratio,
            0.0,
            1.0,
        )
        valid_epoch_mask = (
            np.isfinite(valid_ratio)
            & (valid_ratio > 0.0)
        )
    else:
        valid_epoch_mask = np.ones(
            n_epochs,
            dtype=bool,
        )

    component_available = bool(
        any_series is not None
        or valid_ratio is not None
        or positive_ratio is not None
    )

    features[
        f"caisr_{event_name}_available"
    ] = float(component_available)

    features[
        f"caisr_{event_name}_mean_valid_ratio"
    ] = (
        finite_mean(valid_ratio)
        if valid_ratio is not None
        else float("nan")
    )

    features[
        f"caisr_{event_name}_valid_epoch_fraction"
    ] = (
        float(np.mean(valid_epoch_mask))
        if valid_ratio is not None
        else (
            1.0
            if component_available
            else float("nan")
        )
    )

    if any_series is not None:
        features[
            f"caisr_{event_name}_positive_epoch_fraction_all"
        ] = float(
            np.mean(any_series)
        )

        features[
            f"caisr_{event_name}_positive_epoch_fraction_valid"
        ] = (
            float(
                np.mean(
                    any_series[
                        valid_epoch_mask
                    ]
                )
            )
            if np.any(valid_epoch_mask)
            else float("nan")
        )

        event_onsets = run_count(
            any_series,
            target=True,
        )

        features[
            f"caisr_{event_name}_event_onsets"
        ] = float(event_onsets)

        features[
            f"caisr_{event_name}_event_onsets_per_hour"
        ] = safe_ratio(
            event_onsets,
            duration_hours,
        )

        longest_run = longest_true_run(
            any_series
        )

        features[
            f"caisr_{event_name}_longest_positive_run_min"
        ] = (
            longest_run
            * SECONDS_PER_EPOCH
            / 60.0
        )
    else:
        features[
            f"caisr_{event_name}_positive_epoch_fraction_all"
        ] = float("nan")
        features[
            f"caisr_{event_name}_positive_epoch_fraction_valid"
        ] = float("nan")
        features[
            f"caisr_{event_name}_event_onsets"
        ] = float("nan")
        features[
            f"caisr_{event_name}_event_onsets_per_hour"
        ] = float("nan")
        features[
            f"caisr_{event_name}_longest_positive_run_min"
        ] = float("nan")

    if positive_ratio is not None:
        clipped_positive = np.clip(
            positive_ratio,
            0.0,
            1.0,
        )

        features[
            f"caisr_{event_name}_mean_positive_ratio"
        ] = finite_mean(
            clipped_positive[
                valid_epoch_mask
            ]
        )
        features[
            f"caisr_{event_name}_p90_positive_ratio"
        ] = finite_quantile(
            clipped_positive[
                valid_epoch_mask
            ],
            0.90,
        )
    else:
        features[
            f"caisr_{event_name}_mean_positive_ratio"
        ] = float("nan")
        features[
            f"caisr_{event_name}_p90_positive_ratio"
        ] = float("nan")

    return features


def extract_caisr_features_from_hdf5(
    record_id: str,
    cache_path_string: str,
) -> dict[str, Any]:
    cache_path = Path(
        cache_path_string
    )

    output: dict[str, Any] = {
        "record_id": record_id,
        "cache_path": str(
            cache_path
        ),
        "feature_status": "ok",
        "feature_error": "",
    }

    try:
        with h5py.File(
            cache_path,
            "r",
        ) as handle:
            stored_record_id = str(
                handle.attrs.get(
                    "record_id",
                    record_id,
                )
            )

            if isinstance(
                handle.attrs.get(
                    "record_id",
                    record_id,
                ),
                bytes,
            ):
                stored_record_id = (
                    handle.attrs[
                        "record_id"
                    ].decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            if (
                stored_record_id
                and stored_record_id
                != record_id
            ):
                raise RuntimeError(
                    "HDF5 record ID mismatch: "
                    f"expected {record_id}, found {stored_record_id}"
                )

            n_epochs = int(
                handle.attrs.get(
                    "complete_epoch_count",
                    0,
                )
            )

            if n_epochs <= 0:
                if (
                    "annotations/timing/epoch_index"
                    in handle
                ):
                    n_epochs = int(
                        handle[
                            "annotations/timing/epoch_index"
                        ].shape[0]
                    )

            if n_epochs <= 0:
                raise RuntimeError(
                    "Could not determine a positive epoch count."
                )

            duration_hours = (
                n_epochs
                * SECONDS_PER_EPOCH
                / 3600.0
            )

            output[
                "record_duration_hours"
            ] = duration_hours
            output[
                "record_epoch_count"
            ] = float(n_epochs)

            if "annotations/caisr" not in handle:
                datasets: dict[
                    str,
                    np.ndarray,
                ] = {}
            else:
                datasets = collect_datasets(
                    handle[
                        "annotations/caisr"
                    ]
                )

            file_available_value = find_dataset(
                datasets,
                [
                    "file_available",
                    "available",
                ],
            )

            if file_available_value is None:
                file_available = bool(
                    datasets
                )
            else:
                numeric = as_1d_numeric(
                    file_available_value
                )
                file_available = bool(
                    np.any(
                        np.isfinite(numeric)
                        & (numeric > 0.5)
                    )
                )

            output[
                "caisr_file_available"
            ] = float(
                file_available
            )

            stage = read_numeric_series(
                datasets,
                [
                    "stage",
                    "stage_caisr",
                    "sleep_stage",
                ],
                n_epochs,
            )

            explicit_stage_valid = (
                read_boolean_series(
                    datasets,
                    [
                        "stage_valid",
                        "sleep_stage_valid",
                    ],
                    n_epochs,
                )
            )

            if stage is None:
                stage = np.full(
                    n_epochs,
                    np.nan,
                    dtype=float,
                )

            inferred_stage_valid = (
                np.isfinite(stage)
                & np.isin(
                    stage.astype(
                        int,
                        copy=False,
                    ),
                    list(
                        STAGE_CODES.values()
                    ),
                )
            )

            if explicit_stage_valid is None:
                stage_valid = (
                    inferred_stage_valid
                )
            else:
                stage_valid = (
                    explicit_stage_valid
                    & inferred_stage_valid
                )

            valid_stage_count = int(
                np.sum(
                    stage_valid
                )
            )

            output[
                "caisr_stage_available"
            ] = float(
                valid_stage_count > 0
            )
            output[
                "caisr_stage_valid_epoch_fraction"
            ] = safe_ratio(
                valid_stage_count,
                n_epochs,
            )
            output[
                "caisr_stage_valid_hours"
            ] = (
                valid_stage_count
                * SECONDS_PER_EPOCH
                / 3600.0
            )

            stage_counts = []

            for stage_name, stage_code in (
                STAGE_CODES.items()
            ):
                count = int(
                    np.sum(
                        stage_valid
                        & (
                            stage
                            == stage_code
                        )
                    )
                )

                stage_counts.append(
                    count
                )

                output[
                    f"caisr_stage_{stage_name}_fraction_valid"
                ] = safe_ratio(
                    count,
                    valid_stage_count,
                )

                output[
                    f"caisr_stage_{stage_name}_hours"
                ] = (
                    count
                    * SECONDS_PER_EPOCH
                    / 3600.0
                )

            output[
                "caisr_stage_sleep_fraction_valid"
            ] = (
                1.0
                - output[
                    "caisr_stage_wake_fraction_valid"
                ]
                if np.isfinite(
                    output[
                        "caisr_stage_wake_fraction_valid"
                    ]
                )
                else float("nan")
            )

            output[
                "caisr_stage_distribution_entropy"
            ] = normalized_entropy_from_counts(
                np.asarray(
                    stage_counts,
                    dtype=float,
                )
            )

            consecutive_valid = (
                stage_valid[:-1]
                & stage_valid[1:]
            )

            transition_count = int(
                np.sum(
                    consecutive_valid
                    & (
                        stage[:-1]
                        != stage[1:]
                    )
                )
            )

            consecutive_valid_count = int(
                np.sum(
                    consecutive_valid
                )
            )

            output[
                "caisr_stage_transition_fraction"
            ] = safe_ratio(
                transition_count,
                consecutive_valid_count,
            )

            output[
                "caisr_stage_transitions_per_valid_hour"
            ] = safe_ratio(
                transition_count,
                output[
                    "caisr_stage_valid_hours"
                ],
            )

            valid_indices = np.flatnonzero(
                stage_valid
            )

            non_wake_indices = np.flatnonzero(
                stage_valid
                & (
                    stage
                    != STAGE_CODES[
                        "wake"
                    ]
                )
            )

            rem_indices = np.flatnonzero(
                stage_valid
                & (
                    stage
                    == STAGE_CODES[
                        "rem"
                    ]
                )
            )

            if non_wake_indices.size:
                sleep_onset_index = int(
                    non_wake_indices[0]
                )
                last_sleep_index = int(
                    non_wake_indices[-1]
                )

                output[
                    "caisr_sleep_onset_min"
                ] = (
                    sleep_onset_index
                    * SECONDS_PER_EPOCH
                    / 60.0
                )
                output[
                    "caisr_sleep_onset_fraction_recording"
                ] = safe_ratio(
                    sleep_onset_index,
                    n_epochs,
                )

                after_onset = np.arange(
                    sleep_onset_index,
                    last_sleep_index + 1,
                )

                after_onset_valid = (
                    stage_valid[
                        after_onset
                    ]
                )

                after_onset_wake = (
                    after_onset_valid
                    & (
                        stage[
                            after_onset
                        ]
                        == STAGE_CODES[
                            "wake"
                        ]
                    )
                )

                output[
                    "caisr_waso_fraction_valid_after_onset"
                ] = safe_ratio(
                    int(
                        np.sum(
                            after_onset_wake
                        )
                    ),
                    int(
                        np.sum(
                            after_onset_valid
                        )
                    ),
                )

                wake_sequence = (
                    after_onset_wake
                    & after_onset_valid
                )

                output[
                    "caisr_wake_bouts_after_onset"
                ] = float(
                    run_count(
                        wake_sequence,
                        target=True,
                    )
                )

                output[
                    "caisr_longest_wake_bout_after_onset_min"
                ] = (
                    longest_true_run(
                        wake_sequence
                    )
                    * SECONDS_PER_EPOCH
                    / 60.0
                )

                rem_after_onset = (
                    rem_indices[
                        rem_indices
                        >= sleep_onset_index
                    ]
                )

                if rem_after_onset.size:
                    output[
                        "caisr_rem_latency_from_sleep_onset_min"
                    ] = (
                        int(
                            rem_after_onset[0]
                        )
                        - sleep_onset_index
                    ) * SECONDS_PER_EPOCH / 60.0
                else:
                    output[
                        "caisr_rem_latency_from_sleep_onset_min"
                    ] = float("nan")
            else:
                output[
                    "caisr_sleep_onset_min"
                ] = float("nan")
                output[
                    "caisr_sleep_onset_fraction_recording"
                ] = float("nan")
                output[
                    "caisr_waso_fraction_valid_after_onset"
                ] = float("nan")
                output[
                    "caisr_wake_bouts_after_onset"
                ] = float("nan")
                output[
                    "caisr_longest_wake_bout_after_onset_min"
                ] = float("nan")
                output[
                    "caisr_rem_latency_from_sleep_onset_min"
                ] = float("nan")

            if valid_indices.size:
                valid_stage_sequence = (
                    stage[
                        valid_indices
                    ].astype(
                        int,
                    )
                )

                output[
                    "caisr_stage_run_count"
                ] = float(
                    run_count(
                        valid_stage_sequence
                    )
                )

                output[
                    "caisr_stage_runs_per_valid_hour"
                ] = safe_ratio(
                    run_count(
                        valid_stage_sequence
                    ),
                    output[
                        "caisr_stage_valid_hours"
                    ],
                )
            else:
                output[
                    "caisr_stage_run_count"
                ] = float("nan")
                output[
                    "caisr_stage_runs_per_valid_hour"
                ] = float("nan")

            probability_matrix, probability_sources = (
                extract_stage_probability_matrix(
                    datasets,
                    n_epochs,
                )
            )

            explicit_probability_valid = (
                read_boolean_series(
                    datasets,
                    [
                        "stage_probability_valid",
                        "stage_prob_valid",
                    ],
                    n_epochs,
                )
            )

            if probability_matrix is not None:
                probability_matrix = np.asarray(
                    probability_matrix,
                    dtype=float,
                )

                row_finite = np.all(
                    np.isfinite(
                        probability_matrix
                    ),
                    axis=1,
                )

                row_range_valid = np.all(
                    (
                        probability_matrix
                        >= -1e-6
                    )
                    & (
                        probability_matrix
                        <= 1.0
                        + 1e-6
                    ),
                    axis=1,
                )

                if explicit_probability_valid is None:
                    probability_valid = (
                        row_finite
                        & row_range_valid
                    )
                else:
                    probability_valid = (
                        explicit_probability_valid
                        & row_finite
                        & row_range_valid
                    )

                clipped_probabilities = np.clip(
                    probability_matrix,
                    1e-8,
                    1.0,
                )

                row_sum = np.sum(
                    clipped_probabilities,
                    axis=1,
                )

                row_sum_valid = (
                    probability_valid
                    & np.isfinite(
                        row_sum
                    )
                    & (
                        row_sum
                        > 0
                    )
                )

                normalized_probabilities = np.full_like(
                    clipped_probabilities,
                    np.nan,
                    dtype=float,
                )

                normalized_probabilities[
                    row_sum_valid
                ] = (
                    clipped_probabilities[
                        row_sum_valid
                    ]
                    / row_sum[
                        row_sum_valid,
                        None,
                    ]
                )

                probability_valid = (
                    row_sum_valid
                )

                output[
                    "caisr_stage_probability_available"
                ] = float(
                    np.any(
                        probability_valid
                    )
                )

                output[
                    "caisr_stage_probability_valid_epoch_fraction"
                ] = safe_ratio(
                    int(
                        np.sum(
                            probability_valid
                        )
                    ),
                    n_epochs,
                )

                for column_index, stage_name in enumerate(
                    STAGE_CODES
                ):
                    values = (
                        normalized_probabilities[
                            probability_valid,
                            column_index,
                        ]
                    )

                    output[
                        f"caisr_prob_{stage_name}_mean"
                    ] = finite_mean(
                        values
                    )
                    output[
                        f"caisr_prob_{stage_name}_std"
                    ] = finite_std(
                        values
                    )
                    output[
                        f"caisr_prob_{stage_name}_p10"
                    ] = finite_quantile(
                        values,
                        0.10,
                    )
                    output[
                        f"caisr_prob_{stage_name}_p90"
                    ] = finite_quantile(
                        values,
                        0.90,
                    )

                valid_probabilities = (
                    normalized_probabilities[
                        probability_valid
                    ]
                )

                if (
                    valid_probabilities.ndim
                    == 2
                    and valid_probabilities.shape[0]
                    > 0
                ):
                    maximum_probability = np.max(
                        valid_probabilities,
                        axis=1,
                    )

                    entropy = -np.sum(
                        valid_probabilities
                        * np.log(
                            np.clip(
                                valid_probabilities,
                                1e-8,
                                1.0,
                            )
                        ),
                        axis=1,
                    ) / np.log(
                        valid_probabilities.shape[1]
                    )

                    output[
                        "caisr_prob_max_mean"
                    ] = finite_mean(
                        maximum_probability
                    )
                    output[
                        "caisr_prob_max_std"
                    ] = finite_std(
                        maximum_probability
                    )
                    output[
                        "caisr_prob_max_p10"
                    ] = finite_quantile(
                        maximum_probability,
                        0.10,
                    )
                    output[
                        "caisr_prob_entropy_mean"
                    ] = finite_mean(
                        entropy
                    )
                    output[
                        "caisr_prob_entropy_std"
                    ] = finite_std(
                        entropy
                    )
                    output[
                        "caisr_prob_low_confidence_fraction_lt_0_5"
                    ] = float(
                        np.mean(
                            maximum_probability
                            < 0.5
                        )
                    )
                    output[
                        "caisr_prob_low_confidence_fraction_lt_0_6"
                    ] = float(
                        np.mean(
                            maximum_probability
                            < 0.6
                        )
                    )

                    stage_probability_joint_valid = (
                        probability_valid
                        & stage_valid
                    )

                    if np.any(
                        stage_probability_joint_valid
                    ):
                        predicted_codes = (
                            np.argmax(
                                normalized_probabilities[
                                    stage_probability_joint_valid
                                ],
                                axis=1,
                            )
                            + 1
                        )

                        stored_codes = stage[
                            stage_probability_joint_valid
                        ].astype(
                            int
                        )

                        output[
                            "caisr_prob_argmax_stage_agreement"
                        ] = float(
                            np.mean(
                                predicted_codes
                                == stored_codes
                            )
                        )
                    else:
                        output[
                            "caisr_prob_argmax_stage_agreement"
                        ] = float("nan")
                else:
                    output[
                        "caisr_prob_max_mean"
                    ] = float("nan")
                    output[
                        "caisr_prob_max_std"
                    ] = float("nan")
                    output[
                        "caisr_prob_max_p10"
                    ] = float("nan")
                    output[
                        "caisr_prob_entropy_mean"
                    ] = float("nan")
                    output[
                        "caisr_prob_entropy_std"
                    ] = float("nan")
                    output[
                        "caisr_prob_low_confidence_fraction_lt_0_5"
                    ] = float("nan")
                    output[
                        "caisr_prob_low_confidence_fraction_lt_0_6"
                    ] = float("nan")
                    output[
                        "caisr_prob_argmax_stage_agreement"
                    ] = float("nan")

                output[
                    "caisr_stage_probability_source_count"
                ] = float(
                    len(
                        probability_sources
                    )
                )
            else:
                output[
                    "caisr_stage_probability_available"
                ] = 0.0
                output[
                    "caisr_stage_probability_valid_epoch_fraction"
                ] = float("nan")

                for stage_name in STAGE_CODES:
                    for suffix in [
                        "mean",
                        "std",
                        "p10",
                        "p90",
                    ]:
                        output[
                            f"caisr_prob_{stage_name}_{suffix}"
                        ] = float("nan")

                for name in [
                    "caisr_prob_max_mean",
                    "caisr_prob_max_std",
                    "caisr_prob_max_p10",
                    "caisr_prob_entropy_mean",
                    "caisr_prob_entropy_std",
                    "caisr_prob_low_confidence_fraction_lt_0_5",
                    "caisr_prob_low_confidence_fraction_lt_0_6",
                    "caisr_prob_argmax_stage_agreement",
                ]:
                    output[
                        name
                    ] = float("nan")

                output[
                    "caisr_stage_probability_source_count"
                ] = 0.0

            for event_name in [
                "arousal",
                "respiratory",
                "limb",
            ]:
                output.update(
                    extract_event_features(
                        datasets=datasets,
                        event_name=event_name,
                        n_epochs=n_epochs,
                        duration_hours=duration_hours,
                    )
                )

            output[
                "caisr_annotation_dataset_count"
            ] = float(
                len(
                    datasets
                )
            )

        return output

    except Exception as exception:
        output[
            "feature_status"
        ] = "failed"
        output[
            "feature_error"
        ] = repr(
            exception
        )
        return output


def resolve_cache_path(
    row: pd.Series,
    record_id: str,
    cache_root: Path,
    cache_path_column: str | None,
) -> Path:
    candidates: list[Path] = []

    if cache_path_column is not None:
        value = row.get(
            cache_path_column,
            "",
        )

        if pd.notna(value) and str(value).strip():
            candidates.append(
                Path(
                    str(value)
                ).expanduser()
            )

    candidates.extend(
        [
            cache_root
            / f"{record_id}.h5",
            cache_root
            / f"{record_id}.hdf5",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate HDF5 cache for "
        f"{record_id}. Tried: "
        + ", ".join(
            str(path)
            for path in candidates
        )
    )


def extract_feature_table(
    manifest: pd.DataFrame,
    record_id_column: str,
    cache_path_column: str | None,
    cache_root: Path,
    workers: int,
    progress_every: int,
) -> pd.DataFrame:
    jobs: list[
        tuple[str, str]
    ] = []

    for _, row in manifest.iterrows():
        record_id = clean_identifier(
            row[
                record_id_column
            ]
        )

        cache_path = resolve_cache_path(
            row=row,
            record_id=record_id,
            cache_root=cache_root,
            cache_path_column=cache_path_column,
        )

        jobs.append(
            (
                record_id,
                str(
                    cache_path
                ),
            )
        )

    rows: list[
        dict[str, Any]
    ] = []

    if workers == 1:
        for index, (
            record_id,
            cache_path,
        ) in enumerate(
            jobs,
            start=1,
        ):
            rows.append(
                extract_caisr_features_from_hdf5(
                    record_id,
                    cache_path,
                )
            )

            if (
                index == 1
                or index == len(jobs)
                or index % progress_every == 0
            ):
                print(
                    f"Feature extraction: {index}/{len(jobs)}"
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            future_to_record = {
                executor.submit(
                    extract_caisr_features_from_hdf5,
                    record_id,
                    cache_path,
                ): record_id
                for record_id, cache_path in jobs
            }

            completed = 0

            for future in as_completed(
                future_to_record
            ):
                record_id = (
                    future_to_record[
                        future
                    ]
                )

                try:
                    result = future.result()
                except Exception as exception:
                    result = {
                        "record_id": record_id,
                        "feature_status": "failed",
                        "feature_error": repr(
                            exception
                        ),
                    }

                rows.append(
                    result
                )
                completed += 1

                if (
                    completed == 1
                    or completed == len(jobs)
                    or completed % progress_every == 0
                ):
                    print(
                        f"Feature extraction: {completed}/{len(jobs)}"
                    )

    features = pd.DataFrame(
        rows
    )

    if (
        features["record_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Extracted feature table contains duplicate record IDs."
        )

    features = features.sort_values(
        "record_id"
    ).reset_index(
        drop=True
    )

    return features


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs = {
        "handle_unknown": "ignore",
        "dtype": np.float64,
    }

    try:
        return OneHotEncoder(
            sparse_output=True,
            **kwargs,
        )
    except TypeError:
        return OneHotEncoder(
            sparse=True,
            **kwargs,
        )


def make_numeric_imputer() -> SimpleImputer:
    kwargs = {
        "strategy": "median",
        "add_indicator": True,
    }

    try:
        return SimpleImputer(
            keep_empty_features=True,
            **kwargs,
        )
    except TypeError:
        return SimpleImputer(
            **kwargs,
        )


def build_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
    c_value: float,
    class_weight: str,
    max_iter: int,
    seed: int,
) -> Pipeline:
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    make_numeric_imputer(),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "At least one model feature is required."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )

    classifier = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="liblinear",
        class_weight=(
            "balanced"
            if class_weight == "balanced"
            else None
        ),
        max_iter=max_iter,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# -------------------------------------------------------------------------
# Official PhysioNet Challenge 2026 metric formulas.
# -------------------------------------------------------------------------


def compute_prevalence(
    ages: np.ndarray,
    prevalence_labels: np.ndarray,
    prevalence_ages: np.ndarray,
    gap: float = 0.0,
) -> dict[float, float]:
    unique_ages = np.unique(
        ages[
            np.isfinite(
                ages
            )
        ]
    )

    age_to_labels: dict[
        float,
        list[float],
    ] = defaultdict(
        list
    )

    for age in unique_ages:
        for index in range(
            len(
                prevalence_labels
            )
        ):
            if (
                np.isfinite(
                    prevalence_ages[
                        index
                    ]
                )
                and abs(
                    age
                    - prevalence_ages[
                        index
                    ]
                )
                <= gap
            ):
                age_to_labels[
                    float(
                        age
                    )
                ].append(
                    float(
                        prevalence_labels[
                            index
                        ]
                    )
                )

    age_to_prevalence: dict[
        float,
        float,
    ] = {}

    for age, labels_at_age in (
        age_to_labels.items()
    ):
        if labels_at_age:
            age_to_prevalence[
                age
            ] = (
                max(
                    float(
                        np.sum(
                            labels_at_age
                        )
                    ),
                    0.5,
                )
                / len(
                    labels_at_age
                )
            )

    return age_to_prevalence


def compute_reward(
    labels: np.ndarray,
    binary_predictions: np.ndarray,
    ages: np.ndarray,
    age_to_prevalence: dict[float, float],
) -> float:
    number_of_records = len(
        labels
    )

    scores = np.zeros(
        number_of_records,
        dtype=float,
    )

    number_of_scores = 0

    for index in range(
        number_of_records
    ):
        if not np.isfinite(
            ages[
                index
            ]
        ):
            continue

        age = float(
            ages[
                index
            ]
        )

        if age not in age_to_prevalence:
            continue

        prevalence = age_to_prevalence[
            age
        ]

        prevalence = min(
            max(
                prevalence,
                0.5
                / number_of_records,
            ),
            1.0
            - 0.5
            / number_of_records,
        )

        if (
            labels[
                index
            ] == 1
            and binary_predictions[
                index
            ] == 1
        ):
            scores[
                index
            ] = (
                1.0
                / prevalence
                - 1.0
            )
        elif (
            labels[
                index
            ] == 0
            and binary_predictions[
                index
            ] == 1
        ):
            scores[
                index
            ] = -1.0
        elif (
            labels[
                index
            ] == 1
            and binary_predictions[
                index
            ] == 0
        ):
            scores[
                index
            ] = -1.0
        elif (
            labels[
                index
            ] == 0
            and binary_predictions[
                index
            ] == 0
        ):
            scores[
                index
            ] = (
                1.0
                / (
                    1.0
                    - prevalence
                )
                - 1.0
            )

        number_of_scores += 1

    if number_of_scores == 0:
        return float("nan")

    return float(
        np.sum(
            scores
        )
        / number_of_scores
    )


def compute_age_conditioned_auroc(
    labels: np.ndarray,
    probability_predictions: np.ndarray,
    ages: np.ndarray,
    gap: float = 0.0,
) -> tuple[float, int]:
    positive_indices = np.flatnonzero(
        labels == 1
    )

    negative_indices = np.flatnonzero(
        labels == 0
    )

    numerator = 0.0
    denominator = 0

    for positive_index in positive_indices:
        for negative_index in negative_indices:
            if (
                np.isfinite(
                    ages[
                        positive_index
                    ]
                )
                and np.isfinite(
                    ages[
                        negative_index
                    ]
                )
                and abs(
                    ages[
                        positive_index
                    ]
                    - ages[
                        negative_index
                    ]
                )
                <= gap
            ):
                positive_probability = (
                    probability_predictions[
                        positive_index
                    ]
                )

                negative_probability = (
                    probability_predictions[
                        negative_index
                    ]
                )

                if (
                    positive_probability
                    > negative_probability
                ):
                    numerator += 1.0
                elif (
                    positive_probability
                    == negative_probability
                ):
                    numerator += 0.5

                denominator += 1

    if denominator == 0:
        return float("nan"), 0

    return (
        float(
            numerator
            / denominator
        ),
        int(
            denominator
        ),
    )


def compute_age_weighted_auroc(
    labels: np.ndarray,
    probability_predictions: np.ndarray,
    ages: np.ndarray,
    gap: float = 0.0,
) -> float:
    finite_ages = ages[
        np.isfinite(
            ages
        )
    ]

    if finite_ages.size == 0:
        return float("nan")

    age_grid = np.arange(
        np.min(
            finite_ages
        )
        - gap,
        np.max(
            finite_ages
        )
        + gap
        + 1,
    )

    positive_indices = np.flatnonzero(
        labels == 1
    )

    negative_indices = np.flatnonzero(
        labels == 0
    )

    numerator = np.zeros(
        len(
            age_grid
        ),
        dtype=float,
    )

    denominator = np.zeros(
        len(
            age_grid
        ),
        dtype=float,
    )

    for age_index, age in enumerate(
        age_grid
    ):
        for positive_index in positive_indices:
            for negative_index in negative_indices:
                if (
                    np.isfinite(
                        ages[
                            positive_index
                        ]
                    )
                    and np.isfinite(
                        ages[
                            negative_index
                        ]
                    )
                    and abs(
                        ages[
                            positive_index
                        ]
                        - age
                    )
                    <= gap
                    and abs(
                        ages[
                            negative_index
                        ]
                        - age
                    )
                    <= gap
                ):
                    positive_probability = (
                        probability_predictions[
                            positive_index
                        ]
                    )

                    negative_probability = (
                        probability_predictions[
                            negative_index
                        ]
                    )

                    if (
                        positive_probability
                        > negative_probability
                    ):
                        numerator[
                            age_index
                        ] += 1.0
                    elif (
                        positive_probability
                        == negative_probability
                    ):
                        numerator[
                            age_index
                        ] += 0.5

                    denominator[
                        age_index
                    ] += 1.0

    weights = np.asarray(
        [
            np.sum(
                np.abs(
                    finite_ages
                    - age
                )
                <= gap
            )
            for age in age_grid
        ],
        dtype=float,
    )

    empty = (
        denominator
        == 0
    )

    weights[
        empty
    ] = 0.0
    numerator[
        empty
    ] = 0.0
    denominator[
        empty
    ] = 1.0

    if np.sum(
        weights
    ) == 0:
        return float("nan")

    weights = (
        weights
        / np.sum(
            weights
        )
    )

    return float(
        np.sum(
            weights
            * (
                numerator
                / denominator
            )
        )
    )


def safe_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if np.unique(
        labels
    ).size < 2:
        return float("nan")

    return float(
        roc_auc_score(
            labels,
            probabilities,
        )
    )


def safe_auprc(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if np.sum(
        labels == 1
    ) == 0:
        return float("nan")

    return float(
        average_precision_score(
            labels,
            probabilities,
        )
    )


def compute_binary_metrics(
    labels: np.ndarray,
    binary_predictions: np.ndarray,
) -> dict[str, float | int]:
    matrix = confusion_matrix(
        labels,
        binary_predictions,
        labels=[
            0,
            1,
        ],
    )

    true_negative = int(
        matrix[
            0,
            0,
        ]
    )
    false_positive = int(
        matrix[
            0,
            1,
        ]
    )
    false_negative = int(
        matrix[
            1,
            0,
        ]
    )
    true_positive = int(
        matrix[
            1,
            1,
        ]
    )

    total = (
        true_positive
        + false_positive
        + false_negative
        + true_negative
    )

    accuracy = (
        (
            true_positive
            + true_negative
        )
        / total
        if total > 0
        else float("nan")
    )

    f_denominator = (
        2
        * true_positive
        + false_positive
        + false_negative
    )

    f_measure = (
        2
        * true_positive
        / f_denominator
        if f_denominator > 0
        else float("nan")
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "f_measure": float(
            f_measure
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    binary_predictions: np.ndarray,
    ages: np.ndarray,
    prevalence_labels: np.ndarray,
    prevalence_ages: np.ndarray,
) -> dict[str, float | int]:
    age_to_prevalence = compute_prevalence(
        ages=ages,
        prevalence_labels=prevalence_labels,
        prevalence_ages=prevalence_ages,
        gap=AGE_GAP_YEARS,
    )

    reward = compute_reward(
        labels=labels,
        binary_predictions=binary_predictions,
        ages=ages,
        age_to_prevalence=age_to_prevalence,
    )

    (
        age_conditioned_auroc,
        age_pair_count,
    ) = compute_age_conditioned_auroc(
        labels=labels,
        probability_predictions=probabilities,
        ages=ages,
        gap=AGE_GAP_YEARS,
    )

    age_weighted_auroc = (
        compute_age_weighted_auroc(
            labels=labels,
            probability_predictions=probabilities,
            ages=ages,
            gap=AGE_GAP_YEARS,
        )
    )

    metrics: dict[
        str,
        float | int,
    ] = {
        "reward": reward,
        "age_conditioned_auroc": age_conditioned_auroc,
        "age_weighted_auroc": age_weighted_auroc,
        "auroc": safe_auroc(
            labels,
            probabilities,
        ),
        "auprc": safe_auprc(
            labels,
            probabilities,
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "age_pair_count": age_pair_count,
    }

    metrics.update(
        compute_binary_metrics(
            labels,
            binary_predictions,
        )
    )

    return metrics


def get_feature_names(
    model: Pipeline,
) -> list[str]:
    preprocessor = model.named_steps[
        "preprocessor"
    ]

    try:
        return [
            str(
                value
            )
            for value in (
                preprocessor
                .get_feature_names_out()
            )
        ]
    except Exception:
        return []


def get_coefficients(
    model: Pipeline,
    fold: int,
) -> pd.DataFrame:
    feature_names = get_feature_names(
        model
    )

    coefficients = (
        model.named_steps[
            "classifier"
        ]
        .coef_
        .reshape(
            -1
        )
    )

    if (
        feature_names
        and len(
            feature_names
        )
        != len(
            coefficients
        )
    ):
        raise RuntimeError(
            "Feature-name and coefficient lengths differ."
        )

    if not feature_names:
        feature_names = [
            f"feature_{index}"
            for index in range(
                len(
                    coefficients
                )
            )
        ]

    return pd.DataFrame(
        {
            "fold": fold,
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    )


def json_ready(value: Any) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): json_ready(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            json_ready(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        if not np.isfinite(
            value
        ):
            return None

        return float(
            value
        )

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
    rebuild_features: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    protected_feature_names = {
        "caisr_features_v1.csv",
        "caisr_features_v1.parquet",
        "caisr_feature_extraction_failures.csv",
    }

    existing_model_outputs = [
        path
        for path in output_dir.iterdir()
        if path.name
        not in protected_feature_names
        and path.name
        != ".gitkeep"
    ]

    if (
        existing_model_outputs
        and not overwrite
    ):
        names = ", ".join(
            path.name
            for path in existing_model_outputs[
                :8
            ]
        )

        raise FileExistsError(
            f"Model output directory is not empty: {output_dir}. "
            f"Existing entries include: {names}. "
            "Use --overwrite to replace model outputs."
        )

    if rebuild_features:
        for name in protected_feature_names:
            (
                output_dir
                / name
            ).unlink(
                missing_ok=True
            )


def resolve_demographic_columns(
    frame: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
    dict[str, str],
]:
    numeric_columns: list[
        str
    ] = []

    categorical_columns: list[
        str
    ] = []

    semantic_to_column: dict[
        str,
        str,
    ] = {}

    for semantic_name, candidates in (
        DEMOGRAPHIC_NUMERIC_CANDIDATES.items()
    ):
        column = find_column(
            frame,
            candidates,
            f"the {semantic_name} feature",
            required=True,
        )

        numeric_columns.append(
            column
        )

        semantic_to_column[
            semantic_name
        ] = column

    for semantic_name, candidates in (
        DEMOGRAPHIC_CATEGORICAL_CANDIDATES.items()
    ):
        column = find_column(
            frame,
            candidates,
            f"the {semantic_name} feature",
            required=True,
        )

        categorical_columns.append(
            column
        )

        semantic_to_column[
            semantic_name
        ] = column

    return (
        numeric_columns,
        categorical_columns,
        semantic_to_column,
    )


def run_oof_model(
    frame: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    numeric_columns: list[str],
    categorical_columns: list[str],
    threshold: float,
    c_value: float,
    class_weight: str,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    model_dir = (
        output_dir
        / model_name
    )

    fold_model_dir = (
        model_dir
        / "fold_models"
    )

    fold_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_columns = (
        numeric_columns
        + categorical_columns
    )

    labels = frame[
        "_label"
    ].to_numpy(
        dtype=int
    )

    ages = frame[
        "_age"
    ].to_numpy(
        dtype=float
    )

    prevalence_labels = labels.copy()
    prevalence_ages = ages.copy()

    folds = sorted(
        frame[
            "_fold"
        ].unique().tolist()
    )

    oof_probability = np.full(
        len(
            frame
        ),
        np.nan,
        dtype=float,
    )

    fold_metric_rows: list[
        dict[str, Any]
    ] = []

    coefficient_frames: list[
        pd.DataFrame
    ] = []

    fold_schemas: dict[
        str,
        Any,
    ] = {}

    print(
        f"\n=== {model_name} ==="
    )
    print(
        "Numeric features:",
        len(
            numeric_columns
        ),
    )
    print(
        "Categorical features:",
        categorical_columns,
    )

    for fold in folds:
        validation_mask = (
            frame[
                "_fold"
            ]
            == fold
        )

        training_mask = (
            ~validation_mask
        )

        training_frame = frame.loc[
            training_mask
        ]

        validation_frame = frame.loc[
            validation_mask
        ]

        training_labels = (
            training_frame[
                "_label"
            ].to_numpy(
                dtype=int
            )
        )

        validation_labels = (
            validation_frame[
                "_label"
            ].to_numpy(
                dtype=int
            )
        )

        if np.unique(
            training_labels
        ).size < 2:
            raise ValueError(
                f"Training complement for fold {fold} "
                "does not contain both classes."
            )

        model = build_model(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            c_value=c_value,
            class_weight=class_weight,
            max_iter=max_iter,
            seed=seed,
        )

        model.fit(
            training_frame[
                feature_columns
            ],
            training_labels,
        )

        validation_probability = (
            model.predict_proba(
                validation_frame[
                    feature_columns
                ]
            )[
                :,
                1,
            ]
        )

        validation_binary = (
            validation_probability
            >= threshold
        ).astype(
            int
        )

        validation_positions = np.flatnonzero(
            validation_mask.to_numpy()
        )

        oof_probability[
            validation_positions
        ] = validation_probability

        metrics = evaluate_predictions(
            labels=validation_labels,
            probabilities=validation_probability,
            binary_predictions=validation_binary,
            ages=validation_frame[
                "_age"
            ].to_numpy(
                dtype=float
            ),
            prevalence_labels=prevalence_labels,
            prevalence_ages=prevalence_ages,
        )

        metrics.update(
            {
                "fold": int(
                    fold
                ),
                "train_records": int(
                    len(
                        training_frame
                    )
                ),
                "validation_records": int(
                    len(
                        validation_frame
                    )
                ),
                "train_positives": int(
                    np.sum(
                        training_labels
                        == 1
                    )
                ),
                "validation_positives": int(
                    np.sum(
                        validation_labels
                        == 1
                    )
                ),
                "probability_min": float(
                    np.min(
                        validation_probability
                    )
                ),
                "probability_median": float(
                    np.median(
                        validation_probability
                    )
                ),
                "probability_max": float(
                    np.max(
                        validation_probability
                    )
                ),
            }
        )

        fold_metric_rows.append(
            metrics
        )

        joblib.dump(
            model,
            fold_model_dir
            / f"fold_{fold}.joblib",
            compress=3,
        )

        fold_schemas[
            str(
                fold
            )
        ] = {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "transformed_feature_names": get_feature_names(
                model
            ),
        }

        coefficient_frames.append(
            get_coefficients(
                model,
                int(
                    fold
                ),
            )
        )

        print(
            f"Fold {fold}: "
            f"n={len(validation_frame)}, "
            f"positive={int(np.sum(validation_labels == 1))}, "
            f"age-conditioned AUROC="
            f"{metrics['age_conditioned_auroc']:.4f}, "
            f"AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}"
        )

    if not np.all(
        np.isfinite(
            oof_probability
        )
    ):
        raise RuntimeError(
            "OOF probability coverage is incomplete."
        )

    oof_binary = (
        oof_probability
        >= threshold
    ).astype(
        int
    )

    overall_metrics = evaluate_predictions(
        labels=labels,
        probabilities=oof_probability,
        binary_predictions=oof_binary,
        ages=ages,
        prevalence_labels=prevalence_labels,
        prevalence_ages=prevalence_ages,
    )

    overall_metrics.update(
        {
            "model_name": model_name,
            "records": int(
                len(
                    frame
                )
            ),
            "positives": int(
                np.sum(
                    labels
                    == 1
                )
            ),
            "negatives": int(
                np.sum(
                    labels
                    == 0
                )
            ),
            "threshold": float(
                threshold
            ),
            "numeric_feature_count": int(
                len(
                    numeric_columns
                )
            ),
            "categorical_feature_count": int(
                len(
                    categorical_columns
                )
            ),
        }
    )

    oof_predictions = pd.DataFrame(
        {
            "record_id": frame[
                "_record_id"
            ].to_numpy(),
            "SiteID": frame[
                "_site_id"
            ].to_numpy(),
            "BDSPPatientID": frame[
                "_patient_id"
            ].to_numpy(),
            "fold": frame[
                "_fold"
            ].to_numpy(
                dtype=int
            ),
            "Age": ages,
            "true_Cognitive_Impairment": labels,
            "Cognitive_Impairment": oof_binary,
            "Cognitive_Impairment_Probability": oof_probability,
        }
    )

    official_predictions = (
        oof_predictions[
            [
                "SiteID",
                "BDSPPatientID",
                "Cognitive_Impairment",
                "Cognitive_Impairment_Probability",
            ]
        ]
        .copy()
    )

    site_metric_rows: list[
        dict[str, Any]
    ] = []

    for site, site_frame in (
        oof_predictions.groupby(
            "SiteID",
            sort=True,
        )
    ):
        site_labels = site_frame[
            "true_Cognitive_Impairment"
        ].to_numpy(
            dtype=int
        )

        site_probabilities = site_frame[
            "Cognitive_Impairment_Probability"
        ].to_numpy(
            dtype=float
        )

        site_binary = site_frame[
            "Cognitive_Impairment"
        ].to_numpy(
            dtype=int
        )

        site_ages = site_frame[
            "Age"
        ].to_numpy(
            dtype=float
        )

        site_metrics = evaluate_predictions(
            labels=site_labels,
            probabilities=site_probabilities,
            binary_predictions=site_binary,
            ages=site_ages,
            prevalence_labels=prevalence_labels,
            prevalence_ages=prevalence_ages,
        )

        site_metrics.update(
            {
                "site": site,
                "records": int(
                    len(
                        site_frame
                    )
                ),
                "positives": int(
                    np.sum(
                        site_labels
                        == 1
                    )
                ),
            }
        )

        site_metric_rows.append(
            site_metrics
        )

    fold_metrics = pd.DataFrame(
        fold_metric_rows
    ).sort_values(
        "fold"
    )

    fold_summary_rows = []

    for metric_name in [
        "reward",
        "age_conditioned_auroc",
        "age_weighted_auroc",
        "auroc",
        "auprc",
        "brier_score",
        "accuracy",
        "f_measure",
    ]:
        values = pd.to_numeric(
            fold_metrics[
                metric_name
            ],
            errors="coerce",
        )

        fold_summary_rows.append(
            {
                "metric": metric_name,
                "mean": float(
                    values.mean()
                ),
                "standard_deviation": float(
                    values.std(
                        ddof=1
                    )
                ),
                "minimum": float(
                    values.min()
                ),
                "maximum": float(
                    values.max()
                ),
            }
        )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    coefficient_summary = (
        coefficients.groupby(
            "feature",
            as_index=False,
        )
        .agg(
            folds_present=(
                "fold",
                "nunique",
            ),
            mean_coefficient=(
                "coefficient",
                "mean",
            ),
            standard_deviation=(
                "coefficient",
                "std",
            ),
            mean_absolute_coefficient=(
                "absolute_coefficient",
                "mean",
            ),
        )
        .sort_values(
            "mean_absolute_coefficient",
            ascending=False,
        )
    )

    oof_predictions.to_csv(
        model_dir
        / "oof_predictions.csv",
        index=False,
    )

    official_predictions.to_csv(
        model_dir
        / "oof_predictions_official.csv",
        index=False,
    )

    fold_metrics.to_csv(
        model_dir
        / "fold_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        fold_summary_rows
    ).to_csv(
        model_dir
        / "fold_metrics_summary.csv",
        index=False,
    )

    pd.DataFrame(
        site_metric_rows
    ).sort_values(
        "site"
    ).to_csv(
        model_dir
        / "site_metrics.csv",
        index=False,
    )

    coefficients.to_csv(
        model_dir
        / "fold_feature_coefficients.csv",
        index=False,
    )

    coefficient_summary.to_csv(
        model_dir
        / "feature_coefficient_summary.csv",
        index=False,
    )

    with (
        model_dir
        / "overall_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_ready(
                overall_metrics
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    with (
        model_dir
        / "feature_schema.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_ready(
                {
                    "fold_schemas": fold_schemas,
                }
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    score_lines = [
        f"Reward: {overall_metrics['reward']:.6f}",
        (
            "Age-conditioned AUROC: "
            f"{overall_metrics['age_conditioned_auroc']:.6f}"
        ),
        (
            "Age-weighted AUROC: "
            f"{overall_metrics['age_weighted_auroc']:.6f}"
        ),
        f"AUROC: {overall_metrics['auroc']:.6f}",
        f"AUPRC: {overall_metrics['auprc']:.6f}",
        f"Accuracy: {overall_metrics['accuracy']:.6f}",
        f"F-measure: {overall_metrics['f_measure']:.6f}",
        f"Brier score: {overall_metrics['brier_score']:.6f}",
        (
            "Age-conditioned positive-negative pairs: "
            f"{overall_metrics['age_pair_count']}"
        ),
    ]

    (
        model_dir
        / "scores.txt"
    ).write_text(
        "\n".join(
            score_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\n{model_name} overall OOF:"
    )

    for line in score_lines:
        print(
            line
        )

    return {
        "model_name": model_name,
        "overall_metrics": overall_metrics,
        "oof_predictions": oof_predictions,
    }


def main() -> None:
    args = parse_arguments()

    if args.workers < 1:
        raise ValueError(
            "--workers must be at least 1."
        )

    if args.progress_every < 1:
        raise ValueError(
            "--progress-every must be at least 1."
        )

    if not (
        0.0
        <= args.threshold
        <= 1.0
    ):
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    if args.c <= 0:
        raise ValueError(
            "--c must be positive."
        )

    prepare_output_directory(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        rebuild_features=args.rebuild_features,
    )

    manifest = read_table(
        args.manifest
    ).copy()

    fold_column = find_fold_column(
        manifest
    )

    label_column = find_column(
        manifest,
        LABEL_CANDIDATES,
        "the outcome label",
    )

    age_column = find_column(
        manifest,
        AGE_CANDIDATES,
        "the age column",
    )

    site_column = find_column(
        manifest,
        SITE_CANDIDATES,
        "the site column",
    )

    patient_column = find_column(
        manifest,
        PATIENT_CANDIDATES,
        "the patient identifier",
    )

    record_id_column = find_column(
        manifest,
        RECORD_ID_CANDIDATES,
        "the record identifier",
    )

    cache_path_column = find_column(
        manifest,
        CACHE_PATH_CANDIDATES,
        "the HDF5 cache path",
        required=False,
    )

    (
        demographic_numeric,
        demographic_categorical,
        demographic_semantic_to_column,
    ) = resolve_demographic_columns(
        manifest
    )

    manifest[
        "_label"
    ] = manifest[
        label_column
    ].map(
        parse_binary_label
    )

    manifest[
        "_age"
    ] = pd.to_numeric(
        manifest[
            age_column
        ],
        errors="coerce",
    )

    manifest[
        "_fold"
    ] = pd.to_numeric(
        manifest[
            fold_column
        ],
        errors="raise",
    ).astype(
        int
    )

    manifest[
        "_site_id"
    ] = manifest[
        site_column
    ].map(
        clean_identifier
    )

    manifest[
        "_patient_id"
    ] = manifest[
        patient_column
    ].map(
        clean_identifier
    )

    manifest[
        "_record_id"
    ] = manifest[
        record_id_column
    ].map(
        clean_identifier
    )

    eligible = manifest[
        manifest[
            "_label"
        ].isin(
            [
                0.0,
                1.0,
            ]
        )
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No eligible labeled records were found."
        )

    if (
        eligible[
            "_record_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "The eligible manifest contains duplicate record IDs."
        )

    feature_parquet_path = (
        args.output_dir
        / "caisr_features_v1.parquet"
    )

    feature_csv_path = (
        args.output_dir
        / "caisr_features_v1.csv"
    )

    failure_path = (
        args.output_dir
        / "caisr_feature_extraction_failures.csv"
    )

    print(
        "=== CAISR baseline preflight ==="
    )
    print(
        "Manifest:",
        args.manifest,
    )
    print(
        "Eligible records:",
        len(
            eligible
        ),
    )
    print(
        "Positive records:",
        int(
            eligible[
                "_label"
            ].sum()
        ),
    )
    print(
        "Folds:",
        sorted(
            eligible[
                "_fold"
            ].unique().tolist()
        ),
    )
    print(
        "Feature cache:",
        feature_parquet_path,
    )

    feature_cache_exists = (
        feature_parquet_path.is_file()
        or feature_csv_path.is_file()
    )

    if (
        feature_cache_exists
        and not args.rebuild_features
    ):
        print(
            "Reusing existing CAISR feature table."
        )

        if feature_parquet_path.is_file():
            try:
                features = pd.read_parquet(
                    feature_parquet_path
                )
            except ImportError:
                if not feature_csv_path.is_file():
                    raise
                features = pd.read_csv(
                    feature_csv_path
                )
        else:
            features = pd.read_csv(
                feature_csv_path
            )
    else:
        print(
            "Extracting CAISR features from HDF5."
        )

        features = extract_feature_table(
            manifest=eligible,
            record_id_column=record_id_column,
            cache_path_column=cache_path_column,
            cache_root=args.cache_root,
            workers=args.workers,
            progress_every=args.progress_every,
        )

        failures = features[
            ~features[
                "feature_status"
            ].eq(
                "ok"
            )
        ].copy()

        failures.to_csv(
            failure_path,
            index=False,
        )

        if not failures.empty:
            raise RuntimeError(
                f"CAISR feature extraction failed for "
                f"{len(failures)} records. See: {failure_path}"
            )

        try:
            features.to_parquet(
                feature_parquet_path,
                index=False,
            )
        except ImportError:
            print(
                "Parquet engine unavailable; retaining the CSV feature cache."
            )

        features.to_csv(
            feature_csv_path,
            index=False,
        )

    required_feature_columns = {
        "record_id",
        "feature_status",
    }

    missing_feature_columns = (
        required_feature_columns
        - set(
            features.columns
        )
    )

    if missing_feature_columns:
        raise RuntimeError(
            "Feature table is missing required columns: "
            + ", ".join(
                sorted(
                    missing_feature_columns
                )
            )
        )

    failures = features[
        ~features[
            "feature_status"
        ].eq(
            "ok"
        )
    ].copy()

    failures.to_csv(
        failure_path,
        index=False,
    )

    if not failures.empty:
        raise RuntimeError(
            f"Feature table contains "
            f"{len(failures)} failed records."
        )

    if (
        features[
            "record_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Feature table contains duplicate record IDs."
        )

    features_for_merge = (
        features.drop(
            columns=[
                "cache_path",
                "feature_status",
                "feature_error",
            ],
            errors="ignore",
        )
        .rename(
            columns={
                "record_id": "_feature_record_id",
            }
        )
    )

    modeling = eligible.merge(
        features_for_merge,
        left_on="_record_id",
        right_on="_feature_record_id",
        how="left",
        validate="one_to_one",
    )

    if modeling[
        "_feature_record_id"
    ].isna().any():
        raise RuntimeError(
            "Some eligible records did not match extracted CAISR features."
        )

    caisr_feature_columns = sorted(
        [
            column
            for column in modeling.columns
            if (
                column.startswith(
                    "caisr_"
                )
                or column
                in {
                    "record_duration_hours",
                    "record_epoch_count",
                }
            )
            and pd.api.types.is_numeric_dtype(
                modeling[
                    column
                ]
            )
        ]
    )

    if not caisr_feature_columns:
        raise RuntimeError(
            "No numeric CAISR features were found."
        )

    all_missing_features = [
        column
        for column in caisr_feature_columns
        if pd.to_numeric(
            modeling[
                column
            ],
            errors="coerce",
        ).notna().sum()
        == 0
    ]

    if all_missing_features:
        modeling = modeling.drop(
            columns=all_missing_features
        )

        caisr_feature_columns = [
            column
            for column in caisr_feature_columns
            if column
            not in all_missing_features
        ]

    feature_coverage_rows = []

    for column in caisr_feature_columns:
        values = pd.to_numeric(
            modeling[
                column
            ],
            errors="coerce",
        )

        feature_coverage_rows.append(
            {
                "feature": column,
                "non_missing_records": int(
                    values.notna().sum()
                ),
                "non_missing_fraction": float(
                    values.notna().mean()
                ),
                "mean": (
                    float(
                        values.mean()
                    )
                    if values.notna().any()
                    else float("nan")
                ),
                "standard_deviation": (
                    float(
                        values.std(
                            ddof=0
                        )
                    )
                    if values.notna().any()
                    else float("nan")
                ),
            }
        )

    pd.DataFrame(
        feature_coverage_rows
    ).to_csv(
        args.output_dir
        / "caisr_feature_coverage.csv",
        index=False,
    )

    (
        args.output_dir
        / "excluded_all_missing_features.txt"
    ).write_text(
        "\n".join(
            all_missing_features
        )
        + (
            "\n"
            if all_missing_features
            else ""
        ),
        encoding="utf-8",
    )

    print(
        "Usable CAISR features:",
        len(
            caisr_feature_columns
        ),
    )
    print(
        "Excluded all-missing features:",
        len(
            all_missing_features
        ),
    )
    print(
        "CAISR file availability:",
        (
            pd.to_numeric(
                modeling[
                    "caisr_file_available"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .mean()
        ),
    )

    caisr_only_result = run_oof_model(
        frame=modeling,
        model_name="caisr_only",
        output_dir=args.output_dir,
        numeric_columns=caisr_feature_columns,
        categorical_columns=[],
        threshold=args.threshold,
        c_value=args.c,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    demographics_plus_caisr_result = (
        run_oof_model(
            frame=modeling,
            model_name="demographics_plus_caisr",
            output_dir=args.output_dir,
            numeric_columns=(
                demographic_numeric
                + caisr_feature_columns
            ),
            categorical_columns=demographic_categorical,
            threshold=args.threshold,
            c_value=args.c,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            seed=args.seed,
        )
    )

    comparison_rows = []

    for result in [
        caisr_only_result,
        demographics_plus_caisr_result,
    ]:
        comparison_rows.append(
            result[
                "overall_metrics"
            ]
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        args.output_dir
        / "model_comparison.csv",
        index=False,
    )

    predictions_comparison = (
        caisr_only_result[
            "oof_predictions"
        ][
            [
                "record_id",
                "true_Cognitive_Impairment",
                "Cognitive_Impairment_Probability",
            ]
        ]
        .rename(
            columns={
                "Cognitive_Impairment_Probability": (
                    "caisr_only_probability"
                )
            }
        )
        .merge(
            demographics_plus_caisr_result[
                "oof_predictions"
            ][
                [
                    "record_id",
                    "Cognitive_Impairment_Probability",
                ]
            ].rename(
                columns={
                    "Cognitive_Impairment_Probability": (
                        "demographics_plus_caisr_probability"
                    )
                }
            ),
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
    )

    predictions_comparison.to_csv(
        args.output_dir
        / "oof_prediction_comparison.csv",
        index=False,
    )

    run_metadata = {
        "version": VERSION,
        "manifest": str(
            args.manifest.resolve()
        ),
        "manifest_sha256": sha256_file(
            args.manifest
        ),
        "cache_root": str(
            args.cache_root.resolve()
        ),
        "output_dir": str(
            args.output_dir.resolve()
        ),
        "fold_column": fold_column,
        "label_column": label_column,
        "age_column": age_column,
        "site_column": site_column,
        "patient_column": patient_column,
        "record_id_column": record_id_column,
        "cache_path_column": cache_path_column,
        "demographic_semantic_to_column": (
            demographic_semantic_to_column
        ),
        "caisr_feature_count": len(
            caisr_feature_columns
        ),
        "caisr_features": (
            caisr_feature_columns
        ),
        "excluded_all_missing_features": (
            all_missing_features
        ),
        "threshold": args.threshold,
        "c": args.c,
        "class_weight": args.class_weight,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "age_gap_years": AGE_GAP_YEARS,
    }

    with (
        args.output_dir
        / "run_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_ready(
                run_metadata
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    print(
        "\n=== Model comparison ==="
    )

    print(
        comparison[
            [
                "model_name",
                "age_conditioned_auroc",
                "age_weighted_auroc",
                "auroc",
                "auprc",
                "reward",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved output directory:",
        args.output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(
            f"ERROR: {exception}",
            file=sys.stderr,
        )
        raise
