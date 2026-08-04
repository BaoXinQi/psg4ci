#!/usr/bin/env python3
"""
Structured multimodal PSG fusion experiment.

Research question
-----------------
Does modality-specific representation improve the use of the same complete
PSG input compared with one global PCA, and does a small gated multimodal
fusion block add value beyond balanced concatenation?

No PSG modality is removed. EEG/EOG, Resp/SpO2, ECG, and EMG are retained in
every structured model.

Models
------
demographics
    Demographics only -> MLP.

anchor
    CAISR summary + demographics -> MLP.

global_pca_residual
    All PSG physiological features -> one fold-specific PCA -> protected
    residual correction.

balanced_pca_residual
    Four modality groups -> four independent fold-specific PCAs with equal
    default dimensions -> concatenate -> the same protected residual head.
    This isolates the effect of modality-balanced representation.

balanced_gated_fusion_residual
    The same four independent PCA representations -> small modality-specific
    encoders -> learnable attenuation gates -> shared fusion MLP -> protected
    residual correction. This tests adaptive multimodal fusion while retaining
    all modalities.

All models use the frozen five-fold split. Imputation, variance filtering,
scaling, and PCA are fitted only on the outer-training fold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn


VERSION = "large_structured_baselines_v1"

GROUP_ORDER = (
    "eeg_eog",
    "resp_spo2",
    "ecg",
    "emg",
)

GROUP_PREFIXES = {
    "eeg_eog": ("psg_eeg_", "psg_eog_"),
    "resp_spo2": ("psg_resp_", "psg_spo2_"),
    "ecg": ("psg_ecg_",),
    "emg": ("psg_emg_",),
}

MODEL_NAMES = (
    "demographics",
    "anchor",
    "global_pca_residual",
    "balanced_pca_residual",
    "balanced_gated_fusion_residual",
)

LABEL_CANDIDATES = [
    "demographic__Cognitive_Impairment",
    "Cognitive_Impairment",
    "label",
]
AGE_CANDIDATES = ["demographic__Age", "Age"]
SITE_CANDIDATES = ["demographic__SiteID", "SiteID", "site"]
PATIENT_CANDIDATES = [
    "demographic__BDSPPatientID",
    "BDSPPatientID",
    "subject_id",
    "record_id",
]
RECORD_CANDIDATES = [
    "record_id",
    "BidsFolder",
    "demographic__BidsFolder",
    "subject_id",
]
FOLD_CANDIDATES = ["fold", "cv_fold", "fold_id", "split_fold"]

DEMO_NUMERIC_CANDIDATES = {
    "Age": ["demographic__Age", "Age"],
    "BMI": ["demographic__BMI", "BMI"],
}
DEMO_CATEGORICAL_CANDIDATES = {
    "Sex": ["demographic__Sex", "Sex"],
    "Race": ["demographic__Race", "Race"],
    "Ethnicity": ["demographic__Ethnicity", "Ethnicity"],
}


def parse_args() -> argparse.Namespace:
    root = Path.home() / "fast_data/physionet2026"

    parser = argparse.ArgumentParser(
        description=(
            "Compare global and modality-structured protected PSG residual "
            "fusion while retaining all PSG modalities."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            root
            / "official_small/manifests/full_v1_split_5fold_v1.parquet"
        ),
    )
    parser.add_argument(
        "--summary-features",
        type=Path,
        default=(
            root
            / "official_small/models/baseline_caisr_v1/"
            "caisr_features_v1.parquet"
        ),
    )
    parser.add_argument(
        "--psg-features",
        type=Path,
        default=(
            root
            / "official_small/cache/"
            "psg_physiological_features_v1.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "official_small/models/"
            "psg_structured_multimodal_v1"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20262259, 20262260, 20262261],
    )
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--group-components",
        nargs=4,
        type=int,
        metavar=(
            "EEG_EOG",
            "RESP_SPO2",
            "ECG",
            "EMG",
        ),
        default=[16, 16, 16, 16],
        help=(
            "PCA dimensions for EEG/EOG, Resp/SpO2, ECG, and EMG. "
            "The default retains 64 balanced dimensions in total."
        ),
    )
    parser.add_argument(
        "--global-components",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--minimum-nonmissing-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--residual-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--initial-outer-gate",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--initial-modality-gate",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--residual-penalty",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--modality-gate-penalty",
        type=float,
        default=0.001,
        help=(
            "Weak penalty on gated branch embedding magnitude. "
            "This is an optimization regularizer, not an importance test."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--age-gap-years",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format: {suffix}")


def read_with_csv_fallback(path: Path) -> pd.DataFrame:
    candidates = [path]

    if path.suffix.lower() in {".parquet", ".pq"}:
        candidates.append(path.with_suffix(".csv"))

    errors: list[str] = []

    for candidate in candidates:
        if not candidate.is_file():
            continue

        try:
            return read_table(candidate)
        except Exception as exception:
            errors.append(f"{candidate}: {exception!r}")

    raise FileNotFoundError(
        "Could not read table. "
        + "; ".join(errors or map(str, candidates))
    )


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


def parse_binary(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return float(value)

    if isinstance(
        value,
        (int, np.integer, float, np.floating),
    ):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return float(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "1.0",
        "positive",
    }:
        return 1.0

    if text in {
        "false",
        "f",
        "no",
        "n",
        "0",
        "0.0",
        "negative",
    }:
        return 0.0

    raise ValueError(f"Unrecognized binary label: {value!r}")


def clean_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))

    return str(value).strip()


def prepare_manifest(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_table(path).copy()

    fold_column = find_column(
        frame,
        FOLD_CANDIDATES,
        "the fold column",
        required=False,
    )

    if fold_column is None:
        fuzzy = [
            column
            for column in frame.columns
            if "fold" in str(column).lower()
        ]

        if len(fuzzy) != 1:
            raise ValueError(
                f"Could not identify the fold column: {fuzzy}"
            )

        fold_column = fuzzy[0]

    label_column = find_column(
        frame,
        LABEL_CANDIDATES,
        "the outcome label",
    )
    age_column = find_column(
        frame,
        AGE_CANDIDATES,
        "the age column",
    )
    site_column = find_column(
        frame,
        SITE_CANDIDATES,
        "the site column",
    )
    patient_column = find_column(
        frame,
        PATIENT_CANDIDATES,
        "the patient identifier",
    )
    record_column = find_column(
        frame,
        RECORD_CANDIDATES,
        "the record identifier",
    )

    demographic_numeric: list[str] = []
    demographic_categorical: list[str] = []
    demographic_mapping: dict[str, str] = {}

    for semantic_name, candidates in (
        DEMO_NUMERIC_CANDIDATES.items()
    ):
        column = find_column(
            frame,
            candidates,
            f"the {semantic_name} demographic",
        )
        demographic_numeric.append(column)
        demographic_mapping[semantic_name] = column

    for semantic_name, candidates in (
        DEMO_CATEGORICAL_CANDIDATES.items()
    ):
        column = find_column(
            frame,
            candidates,
            f"the {semantic_name} demographic",
        )
        demographic_categorical.append(column)
        demographic_mapping[semantic_name] = column

    frame["_label"] = frame[label_column].map(parse_binary)
    frame["_age"] = pd.to_numeric(
        frame[age_column],
        errors="coerce",
    )
    frame["_fold"] = pd.to_numeric(
        frame[fold_column],
        errors="raise",
    ).astype(int)
    frame["_site_id"] = frame[site_column].map(clean_id)
    frame["_patient_id"] = frame[patient_column].map(clean_id)
    frame["_record_id"] = frame[record_column].map(clean_id)

    frame = frame[
        frame["_label"].isin([0.0, 1.0])
    ].copy().reset_index(drop=True)

    if frame["_record_id"].duplicated().any():
        raise ValueError("Duplicate manifest record IDs.")

    frame["_row_index"] = np.arange(len(frame), dtype=int)

    metadata = {
        "demographic_numeric": demographic_numeric,
        "demographic_categorical": demographic_categorical,
        "demographic_mapping": demographic_mapping,
    }

    return frame, metadata


def merge_features(
    manifest: pd.DataFrame,
    summary_path: Path,
    psg_path: Path,
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, list[str]],
]:
    summary = read_with_csv_fallback(summary_path).copy()
    psg = read_with_csv_fallback(psg_path).copy()

    for name, frame in [("summary", summary), ("PSG", psg)]:
        if "record_id" not in frame.columns:
            raise ValueError(f"{name} table has no record_id column.")

        frame["record_id"] = frame["record_id"].map(clean_id)

        if frame["record_id"].duplicated().any():
            raise ValueError(f"{name} table has duplicate record IDs.")

    summary = summary.rename(
        columns={"record_id": "_summary_record_id"}
    )
    psg = psg.rename(columns={"record_id": "_psg_record_id"})

    summary = summary.drop(
        columns=[
            "cache_path",
            "feature_status",
            "feature_error",
        ],
        errors="ignore",
    )
    psg = psg.drop(
        columns=["cache_path", "feature_status"],
        errors="ignore",
    )

    frame = manifest.merge(
        summary,
        left_on="_record_id",
        right_on="_summary_record_id",
        how="left",
        validate="one_to_one",
    )

    if frame["_summary_record_id"].isna().any():
        raise RuntimeError(
            "Summary features are missing for "
            f"{int(frame['_summary_record_id'].isna().sum())} records."
        )

    frame = frame.merge(
        psg,
        left_on="_record_id",
        right_on="_psg_record_id",
        how="left",
        validate="one_to_one",
    )

    if frame["_psg_record_id"].isna().any():
        raise RuntimeError(
            "PSG features are missing for "
            f"{int(frame['_psg_record_id'].isna().sum())} records."
        )

    summary_columns = sorted(
        column
        for column in frame.columns
        if (
            (
                column.startswith("caisr_")
                or column
                in {
                    "record_duration_hours",
                    "record_epoch_count",
                }
            )
            and pd.api.types.is_numeric_dtype(frame[column])
            and pd.to_numeric(
                frame[column],
                errors="coerce",
            ).notna().any()
        )
    )

    if not summary_columns:
        raise RuntimeError("No usable CAISR summary features.")

    group_columns: dict[str, list[str]] = {}

    for group_name in GROUP_ORDER:
        prefixes = GROUP_PREFIXES[group_name]

        columns = sorted(
            column
            for column in frame.columns
            if (
                any(
                    column.startswith(prefix)
                    for prefix in prefixes
                )
                and pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).notna().any()
            )
        )

        if not columns:
            raise RuntimeError(
                f"No usable PSG features for group: {group_name}"
            )

        for column in columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

        group_columns[group_name] = columns

    return frame, summary_columns, group_columns


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs = {
        "handle_unknown": "ignore",
        "dtype": np.float32,
    }

    try:
        return OneHotEncoder(
            sparse_output=False,
            **kwargs,
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            **kwargs,
        )


def make_numeric_imputer(
    add_indicator: bool,
) -> SimpleImputer:
    kwargs = {
        "strategy": "median",
        "add_indicator": add_indicator,
    }

    try:
        return SimpleImputer(
            keep_empty_features=True,
            **kwargs,
        )
    except TypeError:
        return SimpleImputer(**kwargs)


def build_tabular_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    add_missing_indicator: bool,
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            make_numeric_imputer(
                                add_missing_indicator
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_columns),
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                list(categorical_columns),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


class FoldPCAProjector:
    def __init__(
        self,
        n_components: int,
        minimum_nonmissing_fraction: float,
        random_state: int,
    ) -> None:
        self.requested_components = int(n_components)
        self.minimum_nonmissing_fraction = float(
            minimum_nonmissing_fraction
        )
        self.random_state = int(random_state)

        self.selected_columns: list[str] = []
        self.imputer: SimpleImputer | None = None
        self.variance_filter: VarianceThreshold | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

        self.input_feature_count = 0
        self.post_variance_feature_count = 0
        self.output_dimension = 0
        self.explained_variance_ratio_sum = np.nan

    def fit(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
    ) -> "FoldPCAProjector":
        minimum_count = max(
            1,
            int(
                math.ceil(
                    len(frame)
                    * self.minimum_nonmissing_fraction
                )
            ),
        )

        self.selected_columns = [
            column
            for column in columns
            if frame[column].notna().sum() >= minimum_count
        ]

        if not self.selected_columns:
            raise RuntimeError(
                "No features survived the nonmissing filter."
            )

        self.input_feature_count = len(self.selected_columns)

        self.imputer = make_numeric_imputer(
            add_indicator=True
        )

        matrix = np.asarray(
            self.imputer.fit_transform(
                frame[self.selected_columns]
            ),
            dtype=np.float32,
        )

        self.variance_filter = VarianceThreshold(
            threshold=1e-8
        )

        matrix = np.asarray(
            self.variance_filter.fit_transform(matrix),
            dtype=np.float32,
        )

        self.post_variance_feature_count = int(
            matrix.shape[1]
        )

        if matrix.shape[1] == 0:
            raise RuntimeError(
                "No features survived variance filtering."
            )

        self.scaler = StandardScaler()

        matrix = np.asarray(
            self.scaler.fit_transform(matrix),
            dtype=np.float32,
        )

        maximum_components = min(
            matrix.shape[0] - 1,
            matrix.shape[1],
        )

        components = min(
            self.requested_components,
            maximum_components,
        )

        if components < 1:
            raise RuntimeError(
                "Could not allocate a PCA component."
            )

        self.pca = PCA(
            n_components=components,
            svd_solver=(
                "randomized"
                if components < min(matrix.shape)
                else "full"
            ),
            random_state=self.random_state,
        )

        self.pca.fit(matrix)

        self.output_dimension = int(components)
        self.explained_variance_ratio_sum = float(
            np.sum(self.pca.explained_variance_ratio_)
        )

        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        if (
            self.imputer is None
            or self.variance_filter is None
            or self.scaler is None
            or self.pca is None
        ):
            raise RuntimeError("Projector has not been fitted.")

        matrix = np.asarray(
            self.imputer.transform(
                frame[self.selected_columns]
            ),
            dtype=np.float32,
        )
        matrix = np.asarray(
            self.variance_filter.transform(matrix),
            dtype=np.float32,
        )
        matrix = np.asarray(
            self.scaler.transform(matrix),
            dtype=np.float32,
        )

        return np.asarray(
            self.pca.transform(matrix),
            dtype=np.float32,
        )


def group_availability(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    present_columns = [
        column
        for column in columns
        if column.endswith("__present")
    ]

    if present_columns:
        present = (
            frame[present_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        return (
            np.max(present, axis=1) > 0.0
        ).astype(np.float32)

    nonmissing = frame[list(columns)].notna().mean(axis=1)

    return (
        nonmissing.to_numpy(dtype=float) > 0.0
    ).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class ProtectedResidualHead(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        dropout: float,
        initial_gate: float,
    ) -> None:
        super().__init__()

        if not 0.0 < initial_gate < 1.0:
            raise ValueError(
                "initial_gate must be strictly between 0 and 1."
            )

        self.feature_network = nn.Sequential(
            nn.Linear(input_dimension, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_layer = nn.Linear(16, 1)

        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        gate_logit = math.log(
            initial_gate / (1.0 - initial_gate)
        )

        self.gate_logit = nn.Parameter(
            torch.tensor(
                gate_logit,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.output_layer(
            self.feature_network(features)
        ).squeeze(-1)

        gate = torch.sigmoid(self.gate_logit)

        return gate * residual, gate


class GatedMultimodalResidual(nn.Module):
    def __init__(
        self,
        input_dimensions: Mapping[str, int],
        dropout: float,
        initial_outer_gate: float,
        initial_modality_gate: float,
        branch_dimension: int = 8,
    ) -> None:
        super().__init__()

        if not 0.0 < initial_outer_gate < 1.0:
            raise ValueError(
                "initial_outer_gate must be in (0, 1)."
            )

        if not 0.0 < initial_modality_gate < 1.0:
            raise ValueError(
                "initial_modality_gate must be in (0, 1)."
            )

        self.group_names = tuple(input_dimensions.keys())
        self.branch_dimension = int(branch_dimension)

        self.branches = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.Linear(
                        input_dimensions[group_name],
                        self.branch_dimension,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for group_name in self.group_names
            }
        )

        modality_gate_logit = math.log(
            initial_modality_gate
            / (1.0 - initial_modality_gate)
        )

        self.modality_gate_logits = nn.ParameterDict(
            {
                group_name: nn.Parameter(
                    torch.tensor(
                        modality_gate_logit,
                        dtype=torch.float32,
                    )
                )
                for group_name in self.group_names
            }
        )

        fusion_dimension = (
            len(self.group_names)
            * self.branch_dimension
            + len(self.group_names)
        )

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dimension, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_layer = nn.Linear(16, 1)

        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        outer_gate_logit = math.log(
            initial_outer_gate
            / (1.0 - initial_outer_gate)
        )

        self.outer_gate_logit = nn.Parameter(
            torch.tensor(
                outer_gate_logit,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        group_features: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        gated_embeddings: list[torch.Tensor] = []
        modality_gates: dict[str, torch.Tensor] = {}

        for group_index, group_name in enumerate(
            self.group_names
        ):
            embedding = self.branches[group_name](
                group_features[group_name]
            )

            gate = torch.sigmoid(
                self.modality_gate_logits[group_name]
            )
            modality_gates[group_name] = gate

            availability_column = availability[
                :,
                group_index:
                group_index + 1,
            ]

            gated_embedding = (
                embedding
                * gate
                * availability_column
            )

            gated_embeddings.append(gated_embedding)

        fusion_input = torch.cat(
            gated_embeddings
            + [availability],
            dim=1,
        )

        residual = self.output_layer(
            self.fusion_network(fusion_input)
        ).squeeze(-1)

        outer_gate = torch.sigmoid(
            self.outer_gate_logit
        )

        correction = outer_gate * residual

        embedding_penalty = torch.mean(
            torch.cat(
                gated_embeddings,
                dim=1,
            )
            ** 2
        )

        return (
            correction,
            outer_gate,
            modality_gates,
            embedding_penalty,
        )


def compute_pos_weight(labels: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))

    if positives == 0:
        raise ValueError("No positive training examples.")

    return float(negatives / positives)


def make_batches(
    count: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch * 100003)

    permutation = torch.randperm(
        count,
        generator=generator,
    ).numpy()

    return [
        permutation[start:start + batch_size]
        for start in range(0, count, batch_size)
    ]


def train_mlp(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    set_seed(seed)

    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=device,
    )
    label_tensor = torch.as_tensor(
        labels,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            compute_pos_weight(labels),
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, args.fixed_epochs + 1):
        model.train()

        total_loss = 0.0
        total_count = 0

        for batch_indices in make_batches(
            len(labels),
            args.batch_size,
            seed,
            epoch,
        ):
            indices = torch.as_tensor(
                batch_indices,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(feature_tensor[indices])

            loss = criterion(
                logits,
                label_tensor[indices],
            )

            loss.backward()

            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip,
                )

            optimizer.step()

            batch_count = int(indices.shape[0])
            total_loss += float(loss.item()) * batch_count
            total_count += batch_count

        history.append(
            {
                "epoch": epoch,
                "training_loss": (
                    total_loss / max(total_count, 1)
                ),
            }
        )

    return pd.DataFrame(history)


def train_protected_residual(
    model: ProtectedResidualHead,
    features: np.ndarray,
    anchor_logits: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    set_seed(seed)

    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=device,
    )
    anchor_tensor = torch.as_tensor(
        anchor_logits,
        dtype=torch.float32,
        device=device,
    )
    label_tensor = torch.as_tensor(
        labels,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            compute_pos_weight(labels),
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.residual_learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, args.fixed_epochs + 1):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_penalty = 0.0
        total_count = 0

        for batch_indices in make_batches(
            len(labels),
            args.batch_size,
            seed,
            epoch,
        ):
            indices = torch.as_tensor(
                batch_indices,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            correction, gate = model(
                feature_tensor[indices]
            )

            logits = (
                anchor_tensor[indices]
                + correction
            )

            bce_loss = criterion(
                logits,
                label_tensor[indices],
            )

            penalty = torch.mean(correction ** 2)

            loss = (
                bce_loss
                + args.residual_penalty
                * penalty
            )

            loss.backward()

            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip,
                )

            optimizer.step()

            batch_count = int(indices.shape[0])

            total_loss += float(loss.item()) * batch_count
            total_bce += float(bce_loss.item()) * batch_count
            total_penalty += float(penalty.item()) * batch_count
            total_count += batch_count

        history.append(
            {
                "epoch": epoch,
                "training_total_loss": (
                    total_loss / max(total_count, 1)
                ),
                "training_bce_loss": (
                    total_bce / max(total_count, 1)
                ),
                "training_residual_penalty": (
                    total_penalty / max(total_count, 1)
                ),
                "outer_gate": float(
                    torch.sigmoid(
                        model.gate_logit
                    ).detach().cpu()
                ),
            }
        )

    return pd.DataFrame(history)


def train_gated_multimodal(
    model: GatedMultimodalResidual,
    group_features: Mapping[str, np.ndarray],
    availability: np.ndarray,
    anchor_logits: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    set_seed(seed)

    group_tensors = {
        group_name: torch.as_tensor(
            values,
            dtype=torch.float32,
            device=device,
        )
        for group_name, values in group_features.items()
    }

    availability_tensor = torch.as_tensor(
        availability,
        dtype=torch.float32,
        device=device,
    )
    anchor_tensor = torch.as_tensor(
        anchor_logits,
        dtype=torch.float32,
        device=device,
    )
    label_tensor = torch.as_tensor(
        labels,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            compute_pos_weight(labels),
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.residual_learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, args.fixed_epochs + 1):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_residual_penalty = 0.0
        total_embedding_penalty = 0.0
        total_count = 0
        final_outer_gate = np.nan
        final_modality_gates: dict[str, float] = {}

        for batch_indices in make_batches(
            len(labels),
            args.batch_size,
            seed,
            epoch,
        ):
            indices = torch.as_tensor(
                batch_indices,
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            (
                correction,
                outer_gate,
                modality_gates,
                embedding_penalty,
            ) = model(
                {
                    group_name: values[indices]
                    for group_name, values
                    in group_tensors.items()
                },
                availability_tensor[indices],
            )

            logits = (
                anchor_tensor[indices]
                + correction
            )

            bce_loss = criterion(
                logits,
                label_tensor[indices],
            )

            residual_penalty = torch.mean(
                correction ** 2
            )

            loss = (
                bce_loss
                + args.residual_penalty
                * residual_penalty
                + args.modality_gate_penalty
                * embedding_penalty
            )

            loss.backward()

            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip,
                )

            optimizer.step()

            batch_count = int(indices.shape[0])

            total_loss += float(loss.item()) * batch_count
            total_bce += float(bce_loss.item()) * batch_count
            total_residual_penalty += (
                float(residual_penalty.item())
                * batch_count
            )
            total_embedding_penalty += (
                float(embedding_penalty.item())
                * batch_count
            )
            total_count += batch_count

            final_outer_gate = float(
                outer_gate.detach().cpu()
            )
            final_modality_gates = {
                group_name: float(
                    gate.detach().cpu()
                )
                for group_name, gate
                in modality_gates.items()
            }

        row: dict[str, float] = {
            "epoch": float(epoch),
            "training_total_loss": (
                total_loss / max(total_count, 1)
            ),
            "training_bce_loss": (
                total_bce / max(total_count, 1)
            ),
            "training_residual_penalty": (
                total_residual_penalty
                / max(total_count, 1)
            ),
            "training_embedding_penalty": (
                total_embedding_penalty
                / max(total_count, 1)
            ),
            "outer_gate": final_outer_gate,
        }

        for group_name, gate in (
            final_modality_gates.items()
        ):
            row[
                f"modality_gate_{group_name}"
            ] = gate

        history.append(row)

    return pd.DataFrame(history)


@torch.no_grad()
def predict_mlp_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=device,
    )

    return model(tensor).cpu().numpy().astype(float)


@torch.no_grad()
def predict_protected_residual(
    model: ProtectedResidualHead,
    features: np.ndarray,
    anchor_logits: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float, np.ndarray]:
    model.eval()

    tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=device,
    )

    correction, gate = model(tensor)
    correction_array = correction.cpu().numpy().astype(float)

    return (
        np.asarray(anchor_logits, dtype=float)
        + correction_array,
        float(gate.detach().cpu()),
        correction_array,
    )


@torch.no_grad()
def predict_gated_multimodal(
    model: GatedMultimodalResidual,
    group_features: Mapping[str, np.ndarray],
    availability: np.ndarray,
    anchor_logits: np.ndarray,
    device: torch.device,
) -> tuple[
    np.ndarray,
    float,
    dict[str, float],
    np.ndarray,
]:
    model.eval()

    group_tensors = {
        group_name: torch.as_tensor(
            values,
            dtype=torch.float32,
            device=device,
        )
        for group_name, values in group_features.items()
    }

    availability_tensor = torch.as_tensor(
        availability,
        dtype=torch.float32,
        device=device,
    )

    (
        correction,
        outer_gate,
        modality_gates,
        _,
    ) = model(
        group_tensors,
        availability_tensor,
    )

    correction_array = correction.cpu().numpy().astype(float)

    return (
        np.asarray(anchor_logits, dtype=float)
        + correction_array,
        float(outer_gate.detach().cpu()),
        {
            group_name: float(
                gate.detach().cpu()
            )
            for group_name, gate
            in modality_gates.items()
        },
        correction_array,
    )


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    output = np.empty_like(logits, dtype=float)

    positive = logits >= 0

    output[positive] = (
        1.0
        / (
            1.0
            + np.exp(-logits[positive])
        )
    )

    exponential = np.exp(logits[~positive])

    output[~positive] = (
        exponential
        / (1.0 + exponential)
    )

    return output


def compute_prevalence(
    ages: np.ndarray,
    prevalence_labels: np.ndarray,
    prevalence_ages: np.ndarray,
    gap: float,
) -> dict[float, float]:
    age_to_labels: dict[float, list[float]] = defaultdict(list)

    for age in np.unique(
        ages[np.isfinite(ages)]
    ):
        for index, label in enumerate(prevalence_labels):
            if (
                np.isfinite(prevalence_ages[index])
                and abs(
                    age - prevalence_ages[index]
                )
                <= gap
            ):
                age_to_labels[float(age)].append(
                    float(label)
                )

    return {
        age: max(float(np.sum(labels)), 0.5) / len(labels)
        for age, labels in age_to_labels.items()
        if labels
    }


def compute_reward(
    labels: np.ndarray,
    predictions: np.ndarray,
    ages: np.ndarray,
    age_to_prevalence: dict[float, float],
) -> float:
    scores: list[float] = []
    n = len(labels)

    for label, prediction, age in zip(
        labels,
        predictions,
        ages,
    ):
        if (
            not np.isfinite(age)
            or float(age) not in age_to_prevalence
        ):
            continue

        prevalence = np.clip(
            age_to_prevalence[float(age)],
            0.5 / n,
            1.0 - 0.5 / n,
        )

        if label == prediction == 1:
            scores.append(1.0 / prevalence - 1.0)
        elif label == prediction == 0:
            scores.append(
                1.0 / (1.0 - prevalence) - 1.0
            )
        else:
            scores.append(-1.0)

    return (
        float(np.mean(scores))
        if scores
        else float("nan")
    )


def compute_age_conditioned_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    ages: np.ndarray,
    gap: float,
) -> tuple[float, int]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)

    positive_probability = probabilities[
        positives
    ][:, None]
    negative_probability = probabilities[
        negatives
    ][None, :]

    age_mask = (
        np.abs(
            ages[positives][:, None]
            - ages[negatives][None, :]
        )
        <= gap
    )

    denominator = int(np.sum(age_mask))

    if denominator == 0:
        return float("nan"), 0

    concordance = (
        (
            positive_probability
            > negative_probability
        ).astype(float)
        + 0.5
        * (
            positive_probability
            == negative_probability
        ).astype(float)
    )

    numerator = float(
        np.sum(
            concordance[age_mask]
        )
    )

    return numerator / denominator, denominator


def compute_age_weighted_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    ages: np.ndarray,
    gap: float,
) -> float:
    finite_ages = ages[np.isfinite(ages)]

    if finite_ages.size == 0:
        return float("nan")

    age_grid = np.arange(
        finite_ages.min() - gap,
        finite_ages.max() + gap + 1,
    )

    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)

    numerator = np.zeros(len(age_grid), dtype=float)
    denominator = np.zeros(len(age_grid), dtype=float)

    for age_index, age in enumerate(age_grid):
        positive_mask = (
            np.abs(
                ages[positives] - age
            )
            <= gap
        )
        negative_mask = (
            np.abs(
                ages[negatives] - age
            )
            <= gap
        )

        selected_positive = positives[
            positive_mask
        ]
        selected_negative = negatives[
            negative_mask
        ]

        if (
            selected_positive.size == 0
            or selected_negative.size == 0
        ):
            continue

        positive_probability = probabilities[
            selected_positive
        ][:, None]
        negative_probability = probabilities[
            selected_negative
        ][None, :]

        numerator[age_index] = float(
            np.sum(
                positive_probability
                > negative_probability
            )
            + 0.5
            * np.sum(
                positive_probability
                == negative_probability
            )
        )
        denominator[age_index] = float(
            selected_positive.size
            * selected_negative.size
        )

    weights = np.asarray(
        [
            np.sum(
                np.abs(finite_ages - age) <= gap
            )
            for age in age_grid
        ],
        dtype=float,
    )

    empty = denominator == 0
    weights[empty] = 0.0
    denominator[empty] = 1.0

    if weights.sum() == 0:
        return float("nan")

    weights /= weights.sum()

    return float(
        np.sum(
            weights
            * numerator
            / denominator
        )
    )


def safe_auroc(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if np.unique(labels).size != 2:
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
    if np.sum(labels == 1) == 0:
        return float("nan")

    return float(
        average_precision_score(
            labels,
            probabilities,
        )
    )


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    f_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    return {
        "accuracy": (
            float(
                (
                    true_positive
                    + true_negative
                )
                / total
            )
            if total
            else float("nan")
        ),
        "f_measure": (
            float(
                2 * true_positive
                / f_denominator
            )
            if f_denominator
            else float("nan")
        ),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
    }


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    ages: np.ndarray,
    threshold: float,
    prevalence_labels: np.ndarray,
    prevalence_ages: np.ndarray,
    age_gap: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    prevalence = compute_prevalence(
        ages=ages,
        prevalence_labels=prevalence_labels,
        prevalence_ages=prevalence_ages,
        gap=age_gap,
    )

    conditioned_auroc, pair_count = (
        compute_age_conditioned_auroc(
            labels=labels,
            probabilities=probabilities,
            ages=ages,
            gap=age_gap,
        )
    )

    metrics: dict[str, float | int] = {
        "reward": compute_reward(
            labels,
            predictions,
            ages,
            prevalence,
        ),
        "age_conditioned_auroc": conditioned_auroc,
        "age_weighted_auroc": compute_age_weighted_auroc(
            labels,
            probabilities,
            ages,
            age_gap,
        ),
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
        "age_pair_count": pair_count,
    }

    metrics.update(
        binary_metrics(
            labels,
            predictions,
        )
    )

    return metrics


def fold_summary(
    fold_frame: pd.DataFrame,
) -> dict[str, float]:
    values = fold_frame[
        "age_conditioned_auroc"
    ]

    return {
        "fold_mean_age_conditioned_auroc": float(
            values.mean()
        ),
        "fold_standard_deviation_age_conditioned_auroc": float(
            values.std(ddof=1)
        ),
        "fold_median_age_conditioned_auroc": float(
            values.median()
        ),
        "fold_minimum_age_conditioned_auroc": float(
            values.min()
        ),
        "fold_maximum_age_conditioned_auroc": float(
            values.max()
        ),
    }


def prepare_output(
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        existing = list(output_dir.iterdir())

        if existing and not overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}. Use --overwrite."
            )

        if overwrite:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def save_oof(
    output_dir: Path,
    frame: pd.DataFrame,
    logits: np.ndarray,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = sigmoid_numpy(logits)

    prediction_frame = pd.DataFrame(
        {
            "record_id": frame["_record_id"],
            "SiteID": frame["_site_id"],
            "BDSPPatientID": frame["_patient_id"],
            "fold": frame["_fold"],
            "Age": frame["_age"],
            "true_Cognitive_Impairment": frame[
                "_label"
            ].astype(int),
            "Cognitive_Impairment": (
                probabilities >= threshold
            ).astype(int),
            "Cognitive_Impairment_Logit": logits,
            "Cognitive_Impairment_Probability": probabilities,
        }
    )

    prediction_frame.to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [json_ready(item) for item in value]

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return value


def main() -> None:
    args = parse_args()

    if args.fixed_epochs < 1:
        raise ValueError("--fixed-epochs must be positive.")

    if min(args.group_components) < 1:
        raise ValueError(
            "All --group-components must be positive."
        )

    if args.global_components < 1:
        raise ValueError(
            "--global-components must be positive."
        )

    if not (
        0.0
        < args.minimum_nonmissing_fraction
        <= 1.0
    ):
        raise ValueError(
            "--minimum-nonmissing-fraction must be in (0, 1]."
        )

    prepare_output(
        args.output_dir,
        args.overwrite,
    )

    device = resolve_device(args.device)

    manifest, metadata = prepare_manifest(
        args.manifest
    )

    (
        frame,
        summary_columns,
        group_columns,
    ) = merge_features(
        manifest,
        args.summary_features,
        args.psg_features,
    )

    group_component_map = {
        group_name: int(component_count)
        for group_name, component_count
        in zip(
            GROUP_ORDER,
            args.group_components,
        )
    }

    global_columns = sorted(
        {
            column
            for columns in group_columns.values()
            for column in columns
        }
    )

    folds = sorted(
        frame["_fold"].unique().tolist()
    )

    labels_all = frame["_label"].to_numpy(dtype=int)
    ages_all = frame["_age"].to_numpy(dtype=float)

    print(
        "=== Structured multimodal PSG preflight ===",
        flush=True,
    )
    print(f"Records: {len(frame)}", flush=True)
    print(
        f"Positives: {int(np.sum(labels_all))}",
        flush=True,
    )
    print(f"Folds: {folds}", flush=True)
    print(
        f"Summary features: {len(summary_columns)}",
        flush=True,
    )

    for group_name in GROUP_ORDER:
        print(
            (
                f"{group_name}: "
                f"raw={len(group_columns[group_name])}, "
                f"requested_pca="
                f"{group_component_map[group_name]}"
            ),
            flush=True,
        )

    print(
        f"Global raw features: {len(global_columns)}",
        flush=True,
    )
    print(
        f"Global requested PCA: {args.global_components}",
        flush=True,
    )
    print(f"Models: {args.models}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Device: {device}", flush=True)

    comparison_rows: list[dict[str, Any]] = []
    logits_by_model: dict[str, list[np.ndarray]] = {
        model_name: []
        for model_name in args.models
    }

    for seed in args.seeds:
        print(
            f"\n=== Seed {seed} ===",
            flush=True,
        )

        seed_logits = {
            model_name: np.full(
                len(frame),
                np.nan,
                dtype=float,
            )
            for model_name in args.models
        }

        fold_rows: dict[str, list[dict[str, Any]]] = {
            model_name: []
            for model_name in args.models
        }

        for outer_fold in folds:
            train_frame = frame[
                frame["_fold"] != outer_fold
            ].copy()
            test_frame = frame[
                frame["_fold"] == outer_fold
            ].copy()

            train_labels = train_frame[
                "_label"
            ].to_numpy(dtype=int)
            test_labels = test_frame[
                "_label"
            ].to_numpy(dtype=int)
            test_ages = test_frame[
                "_age"
            ].to_numpy(dtype=float)
            row_indices = test_frame[
                "_row_index"
            ].to_numpy(dtype=int)

            fold_seed = (
                seed + 1000 * int(outer_fold)
            )

            anchor_numeric = (
                list(summary_columns)
                + metadata["demographic_numeric"]
            )
            anchor_categorical = metadata[
                "demographic_categorical"
            ]
            anchor_columns = (
                anchor_numeric
                + anchor_categorical
            )

            anchor_preprocessor = (
                build_tabular_preprocessor(
                    numeric_columns=anchor_numeric,
                    categorical_columns=anchor_categorical,
                    add_missing_indicator=True,
                )
            )

            anchor_train = np.asarray(
                anchor_preprocessor.fit_transform(
                    train_frame[anchor_columns]
                ),
                dtype=np.float32,
            )
            anchor_test = np.asarray(
                anchor_preprocessor.transform(
                    test_frame[anchor_columns]
                ),
                dtype=np.float32,
            )

            set_seed(fold_seed)

            anchor_model = MLP(
                input_dimension=anchor_train.shape[1],
                dropout=args.dropout,
            ).to(device)

            anchor_history = train_mlp(
                anchor_model,
                anchor_train,
                train_labels,
                args,
                device,
                fold_seed,
            )

            anchor_train_logits = predict_mlp_logits(
                anchor_model,
                anchor_train,
                device,
            )
            anchor_test_logits = predict_mlp_logits(
                anchor_model,
                anchor_test,
                device,
            )

            fold_output = (
                args.output_dir
                / f"seed_{seed}"
                / f"fold_{outer_fold}"
            )
            fold_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            if "demographics" in args.models:
                demographic_numeric = metadata[
                    "demographic_numeric"
                ]
                demographic_categorical = metadata[
                    "demographic_categorical"
                ]
                demographic_columns = (
                    demographic_numeric
                    + demographic_categorical
                )
                demographic_preprocessor = (
                    build_tabular_preprocessor(
                        numeric_columns=demographic_numeric,
                        categorical_columns=demographic_categorical,
                        add_missing_indicator=True,
                    )
                )
                demographic_train = np.asarray(
                    demographic_preprocessor.fit_transform(
                        train_frame[demographic_columns]
                    ),
                    dtype=np.float32,
                )
                demographic_test = np.asarray(
                    demographic_preprocessor.transform(
                        test_frame[demographic_columns]
                    ),
                    dtype=np.float32,
                )
                set_seed(fold_seed)
                demographic_model = MLP(
                    input_dimension=demographic_train.shape[1],
                    dropout=args.dropout,
                ).to(device)
                demographic_history = train_mlp(
                    demographic_model,
                    demographic_train,
                    train_labels,
                    args,
                    device,
                    fold_seed,
                )
                demographic_test_logits = predict_mlp_logits(
                    demographic_model,
                    demographic_test,
                    device,
                )
                seed_logits["demographics"][
                    row_indices
                ] = demographic_test_logits
                demographic_history.to_csv(
                    fold_output
                    / "demographics_training_history.csv",
                    index=False,
                )
                torch.save(
                    {
                        "state_dict": demographic_model.state_dict(),
                        "input_dimension": int(
                            demographic_train.shape[1]
                        ),
                    },
                    fold_output / "demographics_model.pt",
                )
                joblib.dump(
                    demographic_preprocessor,
                    fold_output
                    / "demographics_preprocessor.joblib",
                    compress=3,
                )
                metrics = evaluate(
                    labels=test_labels,
                    probabilities=sigmoid_numpy(
                        demographic_test_logits
                    ),
                    ages=test_ages,
                    threshold=args.threshold,
                    prevalence_labels=labels_all,
                    prevalence_ages=ages_all,
                    age_gap=args.age_gap_years,
                )
                metrics.update(
                    {
                        "model_name": "demographics",
                        "seed": int(seed),
                        "fold": int(outer_fold),
                        "trainable_parameters": int(
                            count_trainable_parameters(
                                demographic_model
                            )
                        ),
                    }
                )
                fold_rows["demographics"].append(metrics)

            anchor_history.to_csv(
                fold_output
                / "anchor_training_history.csv",
                index=False,
            )

            torch.save(
                {
                    "state_dict": anchor_model.state_dict(),
                    "input_dimension": int(
                        anchor_train.shape[1]
                    ),
                },
                fold_output / "anchor_model.pt",
            )

            joblib.dump(
                anchor_preprocessor,
                fold_output
                / "anchor_preprocessor.joblib",
                compress=3,
            )

            if "anchor" in args.models:
                seed_logits["anchor"][
                    row_indices
                ] = anchor_test_logits

                metrics = evaluate(
                    labels=test_labels,
                    probabilities=sigmoid_numpy(
                        anchor_test_logits
                    ),
                    ages=test_ages,
                    threshold=args.threshold,
                    prevalence_labels=labels_all,
                    prevalence_ages=ages_all,
                    age_gap=args.age_gap_years,
                )
                metrics.update(
                    {
                        "model_name": "anchor",
                        "seed": int(seed),
                        "fold": int(outer_fold),
                        "trainable_parameters": int(
                            count_trainable_parameters(
                                anchor_model
                            )
                        ),
                    }
                )
                fold_rows["anchor"].append(metrics)

            need_global = (
                "global_pca_residual"
                in args.models
            )

            need_structured = any(
                model_name in args.models
                for model_name in (
                    "balanced_pca_residual",
                    "balanced_gated_fusion_residual",
                )
            )

            if need_global:
                global_projector = FoldPCAProjector(
                    n_components=args.global_components,
                    minimum_nonmissing_fraction=(
                        args.minimum_nonmissing_fraction
                    ),
                    random_state=fold_seed + 50000,
                ).fit(
                    train_frame,
                    global_columns,
                )

                global_train = global_projector.transform(
                    train_frame
                )
                global_test = global_projector.transform(
                    test_frame
                )

                joblib.dump(
                    global_projector,
                    fold_output
                    / "global_projector.joblib",
                    compress=3,
                )

                global_seed = fold_seed + 100000

                set_seed(global_seed)

                global_model = ProtectedResidualHead(
                    input_dimension=global_train.shape[1],
                    dropout=args.dropout,
                    initial_gate=(
                        args.initial_outer_gate
                    ),
                ).to(device)

                global_history = train_protected_residual(
                    global_model,
                    global_train,
                    anchor_train_logits,
                    train_labels,
                    args,
                    device,
                    global_seed,
                )

                (
                    global_test_logits,
                    global_gate,
                    global_correction,
                ) = predict_protected_residual(
                    global_model,
                    global_test,
                    anchor_test_logits,
                    device,
                )

                seed_logits[
                    "global_pca_residual"
                ][row_indices] = global_test_logits

                metrics = evaluate(
                    labels=test_labels,
                    probabilities=sigmoid_numpy(
                        global_test_logits
                    ),
                    ages=test_ages,
                    threshold=args.threshold,
                    prevalence_labels=labels_all,
                    prevalence_ages=ages_all,
                    age_gap=args.age_gap_years,
                )
                metrics.update(
                    {
                        "model_name": (
                            "global_pca_residual"
                        ),
                        "seed": int(seed),
                        "fold": int(outer_fold),
                        "outer_gate": float(
                            global_gate
                        ),
                        "mean_absolute_residual_logit": float(
                            np.mean(
                                np.abs(
                                    global_correction
                                )
                            )
                        ),
                        "pca_components": int(
                            global_projector.output_dimension
                        ),
                        "pca_explained_variance_ratio_sum": float(
                            global_projector
                            .explained_variance_ratio_sum
                        ),
                        "trainable_parameters": int(
                            count_trainable_parameters(
                                global_model
                            )
                        ),
                    }
                )
                fold_rows[
                    "global_pca_residual"
                ].append(metrics)

                global_history.to_csv(
                    fold_output
                    / (
                        "global_pca_residual_"
                        "training_history.csv"
                    ),
                    index=False,
                )

                torch.save(
                    {
                        "state_dict": (
                            global_model.state_dict()
                        ),
                        "input_dimension": int(
                            global_train.shape[1]
                        ),
                    },
                    fold_output
                    / "global_pca_residual_model.pt",
                )

                del global_model

            if need_structured:
                projectors: dict[
                    str,
                    FoldPCAProjector,
                ] = {}
                structured_train: dict[
                    str,
                    np.ndarray,
                ] = {}
                structured_test: dict[
                    str,
                    np.ndarray,
                ] = {}
                train_availability_columns: list[
                    np.ndarray
                ] = []
                test_availability_columns: list[
                    np.ndarray
                ] = []

                for group_index, group_name in enumerate(
                    GROUP_ORDER
                ):
                    projector = FoldPCAProjector(
                        n_components=(
                            group_component_map[
                                group_name
                            ]
                        ),
                        minimum_nonmissing_fraction=(
                            args.minimum_nonmissing_fraction
                        ),
                        random_state=(
                            fold_seed
                            + 60000
                            + group_index
                            * 1000
                        ),
                    ).fit(
                        train_frame,
                        group_columns[group_name],
                    )

                    projectors[group_name] = projector
                    structured_train[
                        group_name
                    ] = projector.transform(
                        train_frame
                    )
                    structured_test[
                        group_name
                    ] = projector.transform(
                        test_frame
                    )

                    train_availability_columns.append(
                        group_availability(
                            train_frame,
                            group_columns[group_name],
                        )
                    )
                    test_availability_columns.append(
                        group_availability(
                            test_frame,
                            group_columns[group_name],
                        )
                    )

                    joblib.dump(
                        projector,
                        fold_output
                        / (
                            f"{group_name}_"
                            "projector.joblib"
                        ),
                        compress=3,
                    )

                    projector_metadata = {
                        "group_name": group_name,
                        "raw_feature_count": len(
                            group_columns[group_name]
                        ),
                        "selected_feature_count": (
                            projector.input_feature_count
                        ),
                        "post_variance_feature_count": (
                            projector
                            .post_variance_feature_count
                        ),
                        "pca_components": (
                            projector.output_dimension
                        ),
                        "pca_explained_variance_ratio_sum": (
                            projector
                            .explained_variance_ratio_sum
                        ),
                    }

                    (
                        fold_output
                        / (
                            f"{group_name}_"
                            "projector_metadata.json"
                        )
                    ).write_text(
                        json.dumps(
                            json_ready(
                                projector_metadata
                            ),
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )

                train_availability = np.stack(
                    train_availability_columns,
                    axis=1,
                ).astype(np.float32)

                test_availability = np.stack(
                    test_availability_columns,
                    axis=1,
                ).astype(np.float32)

                balanced_train = np.concatenate(
                    [
                        structured_train[group_name]
                        for group_name in GROUP_ORDER
                    ],
                    axis=1,
                )
                balanced_test = np.concatenate(
                    [
                        structured_test[group_name]
                        for group_name in GROUP_ORDER
                    ],
                    axis=1,
                )

                if (
                    "balanced_pca_residual"
                    in args.models
                ):
                    balanced_seed = (
                        fold_seed + 200000
                    )

                    set_seed(balanced_seed)

                    balanced_model = ProtectedResidualHead(
                        input_dimension=(
                            balanced_train.shape[1]
                        ),
                        dropout=args.dropout,
                        initial_gate=(
                            args.initial_outer_gate
                        ),
                    ).to(device)

                    balanced_history = (
                        train_protected_residual(
                            balanced_model,
                            balanced_train,
                            anchor_train_logits,
                            train_labels,
                            args,
                            device,
                            balanced_seed,
                        )
                    )

                    (
                        balanced_test_logits,
                        balanced_gate,
                        balanced_correction,
                    ) = predict_protected_residual(
                        balanced_model,
                        balanced_test,
                        anchor_test_logits,
                        device,
                    )

                    seed_logits[
                        "balanced_pca_residual"
                    ][row_indices] = (
                        balanced_test_logits
                    )

                    metrics = evaluate(
                        labels=test_labels,
                        probabilities=sigmoid_numpy(
                            balanced_test_logits
                        ),
                        ages=test_ages,
                        threshold=args.threshold,
                        prevalence_labels=labels_all,
                        prevalence_ages=ages_all,
                        age_gap=args.age_gap_years,
                    )
                    metrics.update(
                        {
                            "model_name": (
                                "balanced_pca_residual"
                            ),
                            "seed": int(seed),
                            "fold": int(outer_fold),
                            "outer_gate": float(
                                balanced_gate
                            ),
                            "mean_absolute_residual_logit": float(
                                np.mean(
                                    np.abs(
                                        balanced_correction
                                    )
                                )
                            ),
                            "total_pca_components": int(
                                balanced_train.shape[1]
                            ),
                            "trainable_parameters": int(
                                count_trainable_parameters(
                                    balanced_model
                                )
                            ),
                        }
                    )
                    fold_rows[
                        "balanced_pca_residual"
                    ].append(metrics)

                    balanced_history.to_csv(
                        fold_output
                        / (
                            "balanced_pca_residual_"
                            "training_history.csv"
                        ),
                        index=False,
                    )

                    torch.save(
                        {
                            "state_dict": (
                                balanced_model.state_dict()
                            ),
                            "input_dimension": int(
                                balanced_train.shape[1]
                            ),
                        },
                        fold_output
                        / (
                            "balanced_pca_residual_"
                            "model.pt"
                        ),
                    )

                    del balanced_model

                if (
                    "balanced_gated_fusion_residual"
                    in args.models
                ):
                    gated_seed = (
                        fold_seed + 300000
                    )

                    set_seed(gated_seed)

                    gated_model = GatedMultimodalResidual(
                        input_dimensions={
                            group_name: int(
                                structured_train[
                                    group_name
                                ].shape[1]
                            )
                            for group_name in GROUP_ORDER
                        },
                        dropout=args.dropout,
                        initial_outer_gate=(
                            args.initial_outer_gate
                        ),
                        initial_modality_gate=(
                            args.initial_modality_gate
                        ),
                    ).to(device)

                    gated_history = train_gated_multimodal(
                        gated_model,
                        structured_train,
                        train_availability,
                        anchor_train_logits,
                        train_labels,
                        args,
                        device,
                        gated_seed,
                    )

                    (
                        gated_test_logits,
                        gated_outer_gate,
                        gated_modality_gates,
                        gated_correction,
                    ) = predict_gated_multimodal(
                        gated_model,
                        structured_test,
                        test_availability,
                        anchor_test_logits,
                        device,
                    )

                    seed_logits[
                        "balanced_gated_fusion_residual"
                    ][row_indices] = gated_test_logits

                    metrics = evaluate(
                        labels=test_labels,
                        probabilities=sigmoid_numpy(
                            gated_test_logits
                        ),
                        ages=test_ages,
                        threshold=args.threshold,
                        prevalence_labels=labels_all,
                        prevalence_ages=ages_all,
                        age_gap=args.age_gap_years,
                    )
                    metrics.update(
                        {
                            "model_name": (
                                "balanced_gated_fusion_residual"
                            ),
                            "seed": int(seed),
                            "fold": int(outer_fold),
                            "outer_gate": float(
                                gated_outer_gate
                            ),
                            "mean_absolute_residual_logit": float(
                                np.mean(
                                    np.abs(
                                        gated_correction
                                    )
                                )
                            ),
                            **{
                                (
                                    "modality_gate_"
                                    f"{group_name}"
                                ): float(gate)
                                for group_name, gate
                                in gated_modality_gates.items()
                            },
                            "trainable_parameters": int(
                                count_trainable_parameters(
                                    gated_model
                                )
                            ),
                        }
                    )
                    fold_rows[
                        "balanced_gated_fusion_residual"
                    ].append(metrics)

                    gated_history.to_csv(
                        fold_output
                        / (
                            "balanced_gated_fusion_"
                            "training_history.csv"
                        ),
                        index=False,
                    )

                    torch.save(
                        {
                            "state_dict": (
                                gated_model.state_dict()
                            ),
                            "input_dimensions": {
                                group_name: int(
                                    structured_train[
                                        group_name
                                    ].shape[1]
                                )
                                for group_name in GROUP_ORDER
                            },
                            "group_order": list(
                                GROUP_ORDER
                            ),
                        },
                        fold_output
                        / (
                            "balanced_gated_fusion_"
                            "model.pt"
                        ),
                    )

                    del gated_model

            del anchor_model

            if "demographics" in args.models:
                del demographic_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

            fold_message = [f"Fold {outer_fold}"]

            for model_name in args.models:
                latest = fold_rows[model_name][-1]
                fold_message.append(
                    (
                        f"{model_name}="
                        f"{latest['age_conditioned_auroc']:.4f}"
                    )
                )

            print(
                ", ".join(fold_message),
                flush=True,
            )

        for model_name in args.models:
            logits = seed_logits[model_name]

            if not np.all(np.isfinite(logits)):
                raise RuntimeError(
                    "Incomplete OOF logits for "
                    f"{model_name}, seed {seed}."
                )

            model_output = (
                args.output_dir
                / model_name
                / f"seed_{seed}"
            )
            model_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            fold_frame = pd.DataFrame(
                fold_rows[model_name]
            ).sort_values("fold")

            summary = fold_summary(
                fold_frame
            )

            overall = evaluate(
                labels=labels_all,
                probabilities=sigmoid_numpy(logits),
                ages=ages_all,
                threshold=args.threshold,
                prevalence_labels=labels_all,
                prevalence_ages=ages_all,
                age_gap=args.age_gap_years,
            )

            comparison_row = {
                "model_name": model_name,
                "seed": int(seed),
                **overall,
                **summary,
            }

            for column in fold_frame.columns:
                if (
                    column == "outer_gate"
                    or column.startswith(
                        "modality_gate_"
                    )
                    or column
                    == "mean_absolute_residual_logit"
                ):
                    comparison_row[
                        f"mean_{column}"
                    ] = float(
                        fold_frame[column].mean()
                    )

            comparison_rows.append(
                comparison_row
            )

            logits_by_model[
                model_name
            ].append(logits.copy())

            fold_frame.to_csv(
                model_output / "fold_metrics.csv",
                index=False,
            )

            save_oof(
                model_output,
                frame,
                logits,
                args.threshold,
            )

            (
                model_output
                / "overall_metrics.json"
            ).write_text(
                json.dumps(
                    json_ready(
                        comparison_row
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            print(
                (
                    f"{model_name}, seed={seed}: "
                    "pooled AC-AUROC="
                    f"{overall['age_conditioned_auroc']:.6f}, "
                    "fold mean="
                    f"{summary['fold_mean_age_conditioned_auroc']:.6f}, "
                    "median="
                    f"{summary['fold_median_age_conditioned_auroc']:.6f}, "
                    "AUROC="
                    f"{overall['auroc']:.6f}, "
                    "AUPRC="
                    f"{overall['auprc']:.6f}"
                ),
                flush=True,
            )

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        args.output_dir
        / "model_comparison_by_seed.csv",
        index=False,
    )

    ensemble_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        stacked_logits = np.stack(
            logits_by_model[model_name],
            axis=0,
        )

        for ensemble_type in (
            "probability",
            "logit",
        ):
            if ensemble_type == "probability":
                probabilities = np.mean(
                    sigmoid_numpy(
                        stacked_logits
                    ),
                    axis=0,
                )

                logits = np.log(
                    np.clip(
                        probabilities,
                        1e-7,
                        1.0 - 1e-7,
                    )
                    / np.clip(
                        1.0 - probabilities,
                        1e-7,
                        1.0,
                    )
                )
            else:
                logits = np.mean(
                    stacked_logits,
                    axis=0,
                )
                probabilities = sigmoid_numpy(
                    logits
                )

            output = (
                args.output_dir
                / model_name
                / f"ensemble_{ensemble_type}"
            )
            output.mkdir(
                parents=True,
                exist_ok=True,
            )

            overall = evaluate(
                labels=labels_all,
                probabilities=probabilities,
                ages=ages_all,
                threshold=args.threshold,
                prevalence_labels=labels_all,
                prevalence_ages=ages_all,
                age_gap=args.age_gap_years,
            )

            ensemble_row = {
                "model_name": model_name,
                "ensemble_type": ensemble_type,
                "seed_count": int(
                    len(args.seeds)
                ),
                **overall,
            }

            ensemble_rows.append(
                ensemble_row
            )

            save_oof(
                output,
                frame,
                logits,
                args.threshold,
            )

            (
                output
                / "overall_metrics.json"
            ).write_text(
                json.dumps(
                    json_ready(
                        ensemble_row
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    ensemble_frame = pd.DataFrame(
        ensemble_rows
    )

    ensemble_frame.to_csv(
        args.output_dir
        / "model_comparison_seed_ensemble.csv",
        index=False,
    )

    metadata_output = {
        "version": VERSION,
        "research_question": (
            "Does modality-specific balanced representation and gated "
            "fusion improve use of the same complete PSG input compared "
            "with one global PCA?"
        ),
        "manifest": str(args.manifest.resolve()),
        "summary_features": str(
            args.summary_features.resolve()
        ),
        "psg_features": str(
            args.psg_features.resolve()
        ),
        "models": args.models,
        "seeds": args.seeds,
        "fixed_epochs": args.fixed_epochs,
        "group_components": group_component_map,
        "global_components": args.global_components,
        "minimum_nonmissing_fraction": (
            args.minimum_nonmissing_fraction
        ),
        "initial_outer_gate": args.initial_outer_gate,
        "initial_modality_gate": (
            args.initial_modality_gate
        ),
        "residual_penalty": args.residual_penalty,
        "modality_gate_penalty": (
            args.modality_gate_penalty
        ),
        "summary_feature_count": len(
            summary_columns
        ),
        "group_feature_counts": {
            group_name: len(
                group_columns[group_name]
            )
            for group_name in GROUP_ORDER
        },
        "all_modalities_retained": True,
        "excluded_generic_prefixes": [
            "psg_quality__",
            "psg_record__",
        ],
        "demographic_mapping": metadata[
            "demographic_mapping"
        ],
    }

    (
        args.output_dir
        / "run_metadata.json"
    ).write_text(
        json.dumps(
            json_ready(
                metadata_output
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    display_columns = [
        "model_name",
        "seed",
        "age_conditioned_auroc",
        "fold_mean_age_conditioned_auroc",
        "fold_standard_deviation_age_conditioned_auroc",
        "fold_median_age_conditioned_auroc",
        "auroc",
        "auprc",
        "reward",
    ]

    print(
        "\n=== Model comparison by seed ===",
        flush=True,
    )
    print(
        comparison[display_columns].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\n=== Seed-ensemble comparison ===",
        flush=True,
    )
    print(
        ensemble_frame.to_string(index=False),
        flush=True,
    )

    print(
        f"\nSaved: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(
            f"ERROR: {exception}",
            file=sys.stderr,
            flush=True,
        )
        raise
