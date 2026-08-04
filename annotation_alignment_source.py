from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import pyedflib


ROOT = Path.home() / "fast_data/physionet2026/official_small"
DATA_ROOT = ROOT / "data/full"
AUDIT = ROOT / "audit"
MANIFEST_DIR = ROOT / "manifests"

MANIFEST_FILE = MANIFEST_DIR / "record_manifest.csv"
PSG_RECORD_FILE = AUDIT / "psg_record_headers.csv"

PARQUET_OUTPUT = MANIFEST_DIR / "epoch_alignment_30s_v1.parquet"
CSV_OUTPUT = MANIFEST_DIR / "epoch_alignment_30s_v1.csv.gz"

RECORD_SUMMARY_OUTPUT = (
    AUDIT / "epoch_alignment_30s_v1_record_summary.csv"
)

SITE_SUMMARY_OUTPUT = (
    AUDIT / "epoch_alignment_30s_v1_site_summary.csv"
)

EVENT_SUMMARY_OUTPUT = (
    AUDIT / "epoch_alignment_30s_v1_event_summary.csv"
)

ERROR_OUTPUT = (
    AUDIT / "epoch_alignment_30s_v1_errors.csv"
)

EPOCH_SEC = 30.0
VALID_STAGE_CODES = np.array([1, 2, 3, 4, 5])


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return (
        str(value).strip().lower()
        in {"true", "1", "yes", "y"}
    )


def normalize_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def label_semantic(label: str) -> str:
    text = normalize_label(label)

    if "stage" in text:
        return "stage"

    if "arous" in text:
        return "arousal"

    if (
        "limb" in text
        or "plm" in text
        or "leg_movement" in text
    ):
        return "limb"

    if (
        "resp" in text
        or "apnea" in text
        or "hypop" in text
    ):
        return "respiratory"

    return "other"


def find_label(
    labels: list[str],
    exact_candidates: list[str],
    semantic: str | None = None,
    exclude_probability: bool = False,
) -> str | None:
    normalized_to_original = {
        normalize_label(label): label
        for label in labels
    }

    for candidate in exact_candidates:
        normalized = normalize_label(candidate)

        if normalized in normalized_to_original:
            return normalized_to_original[normalized]

    if semantic is None:
        return None

    candidates = []

    for label in labels:
        normalized = normalize_label(label)

        if exclude_probability and "prob" in normalized:
            continue

        if label_semantic(label) == semantic:
            candidates.append(label)

    if not candidates:
        return None

    return sorted(candidates)[0]


def read_annotation_edf(
    path: Path,
) -> tuple[
    list[str],
    dict[str, np.ndarray],
    dict[str, float],
]:
    reader = pyedflib.EdfReader(str(path))

    try:
        labels = [
            str(label).strip()
            for label in reader.getSignalLabels()
        ]

        if len(labels) != len(set(labels)):
            raise RuntimeError(
                f"Duplicate channel labels in {path}"
            )

        sampling_rates = np.asarray(
            reader.getSampleFrequencies(),
            dtype=float,
        )

        signals = {}
        rates = {}

        for index, label in enumerate(labels):
            signals[label] = np.asarray(
                reader.readSignal(index),
                dtype=float,
            )

            rates[label] = float(
                sampling_rates[index]
            )

    finally:
        reader.close()

    return labels, signals, rates


def samples_per_epoch(
    sampling_rate: float,
) -> int:
    value = EPOCH_SEC * sampling_rate
    rounded = int(round(value))

    if rounded < 1:
        raise ValueError(
            f"Invalid sampling rate: {sampling_rate}"
        )

    if not np.isclose(
        value,
        rounded,
        atol=1e-4,
    ):
        raise ValueError(
            "Sampling rate does not map cleanly to "
            f"30-second epochs: {sampling_rate}"
        )

    return rounded


def make_epoch_matrix(
    values: np.ndarray,
    sampling_rate: float,
    n_epochs: int,
) -> np.ndarray:
    per_epoch = samples_per_epoch(
        sampling_rate
    )

    target_samples = n_epochs * per_epoch

    padded = np.full(
        target_samples,
        np.nan,
        dtype=float,
    )

    usable_samples = min(
        target_samples,
        values.size,
    )

    if usable_samples:
        padded[:usable_samples] = values[
            :usable_samples
        ]

    return padded.reshape(
        n_epochs,
        per_epoch,
    )


def aggregate_stage(
    values: np.ndarray,
    sampling_rate: float,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = make_epoch_matrix(
        values,
        sampling_rate,
        n_epochs,
    )

    stage = np.full(
        n_epochs,
        np.nan,
        dtype=float,
    )

    for epoch_index in range(n_epochs):
        row = matrix[epoch_index]
        finite = row[np.isfinite(row)]

        if finite.size == 0:
            continue

        rounded = np.rint(finite).astype(int)

        unique, counts = np.unique(
            rounded,
            return_counts=True,
        )

        stage[epoch_index] = unique[
            np.argmax(counts)
        ]

    valid = np.isin(
        stage,
        VALID_STAGE_CODES,
    )

    stage[~valid] = np.nan

    return stage, valid


def aggregate_binary_event(
    values: np.ndarray,
    sampling_rate: float,
    n_epochs: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    matrix = make_epoch_matrix(
        values,
        sampling_rate,
        n_epochs,
    )

    valid = (
        np.isclose(matrix, 0.0)
        | np.isclose(matrix, 1.0)
    )

    valid_count = valid.sum(axis=1)

    valid_ratio = (
        valid_count
        / matrix.shape[1]
    ).astype(float)

    positive_count = (
        valid
        & np.isclose(matrix, 1.0)
    ).sum(axis=1)

    positive_fraction = np.full(
        n_epochs,
        np.nan,
        dtype=float,
    )

    event_any = np.full(
        n_epochs,
        np.nan,
        dtype=float,
    )

    has_valid = valid_count > 0

    positive_fraction[has_valid] = (
        positive_count[has_valid]
        / valid_count[has_valid]
    )

    event_any[has_valid] = (
        positive_count[has_valid] > 0
    ).astype(float)

    return (
        positive_fraction,
        valid_ratio,
        event_any,
    )


def aggregate_probability(
    values: np.ndarray,
    sampling_rate: float,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = make_epoch_matrix(
        values,
        sampling_rate,
        n_epochs,
    )

    valid = (
        np.isfinite(matrix)
        & (matrix >= 0.0)
        & (matrix <= 1.0)
    )

    valid_count = valid.sum(axis=1)

    valid_ratio = (
        valid_count
        / matrix.shape[1]
    ).astype(float)

    probability_mean = np.full(
        n_epochs,
        np.nan,
        dtype=float,
    )

    has_valid = valid_count > 0

    if has_valid.any():
        safe_values = np.where(
            valid,
            matrix,
            0.0,
        )

        probability_mean[has_valid] = (
            safe_values[has_valid].sum(axis=1)
            / valid_count[has_valid]
        )

    return probability_mean, valid_ratio


def empty_float(n_epochs: int) -> np.ndarray:
    return np.full(
        n_epochs,
        np.nan,
        dtype=float,
    )


def empty_bool(n_epochs: int) -> np.ndarray:
    return np.zeros(
        n_epochs,
        dtype=bool,
    )


def extract_annotation_source(
    path: Path | None,
    source: str,
    n_epochs: int,
) -> tuple[
    dict[str, np.ndarray],
    list[str],
]:
    output: dict[str, np.ndarray] = {}
    detected_labels: list[str] = []

    output[f"{source}_file_available"] = np.full(
        n_epochs,
        path is not None,
        dtype=bool,
    )

    output[f"{source}_stage"] = empty_float(
        n_epochs
    )

    output[f"{source}_stage_valid"] = empty_bool(
        n_epochs
    )

    for event in [
        "arousal",
        "respiratory",
        "limb",
    ]:
        output[
            f"{source}_{event}_positive_fraction"
        ] = empty_float(n_epochs)

        output[
            f"{source}_{event}_valid_ratio"
        ] = empty_float(n_epochs)

        output[
            f"{source}_{event}_any"
        ] = empty_float(n_epochs)

    if source == "caisr":
        for stage_name in [
            "n1",
            "n2",
            "n3",
            "rem",
            "wake",
        ]:
            output[
                f"caisr_prob_{stage_name}"
            ] = empty_float(n_epochs)

        output[
            "caisr_stage_probability_valid"
        ] = empty_bool(n_epochs)

        output[
            "caisr_stage_probability_sum_error"
        ] = empty_float(n_epochs)

        output[
            "caisr_stage_probability_max"
        ] = empty_float(n_epochs)

        output[
            "caisr_stage_probability_entropy"
        ] = empty_float(n_epochs)

        output[
            "caisr_prob_arousal"
        ] = empty_float(n_epochs)

        output[
            "caisr_prob_no_arousal"
        ] = empty_float(n_epochs)

        output[
            "caisr_arousal_probability_valid_ratio"
        ] = empty_float(n_epochs)

        output[
            "caisr_arousal_probability_sum_error"
        ] = empty_float(n_epochs)

    if path is None:
        return output, detected_labels

    labels, signals, rates = read_annotation_edf(
        path
    )

    detected_labels = labels

    if source == "human":
        stage_exact = ["stage_expert"]

        event_exact = {
            "arousal": ["arousal_expert"],
            "respiratory": [
                "resp_expert",
                "respiratory_expert",
            ],
            "limb": [
                "limb_expert",
                "limb_movement_expert",
            ],
        }
    else:
        stage_exact = ["stage_caisr"]

        event_exact = {
            "arousal": ["arousal_caisr"],
            "respiratory": ["resp_caisr"],
            "limb": ["limb_caisr"],
        }

    stage_label = find_label(
        labels,
        exact_candidates=stage_exact,
        semantic="stage",
        exclude_probability=True,
    )

    if stage_label is not None:
        stage, valid = aggregate_stage(
            signals[stage_label],
            rates[stage_label],
            n_epochs,
        )

        output[f"{source}_stage"] = stage
        output[f"{source}_stage_valid"] = valid

    for event in [
        "arousal",
        "respiratory",
        "limb",
    ]:
        event_label = find_label(
            labels,
            exact_candidates=event_exact[event],
            semantic=event,
            exclude_probability=True,
        )

        if event_label is None:
            continue

        (
            positive_fraction,
            valid_ratio,
            event_any,
        ) = aggregate_binary_event(
            signals[event_label],
            rates[event_label],
            n_epochs,
        )

        output[
            f"{source}_{event}_positive_fraction"
        ] = positive_fraction

        output[
            f"{source}_{event}_valid_ratio"
        ] = valid_ratio

        output[
            f"{source}_{event}_any"
        ] = event_any

    if source == "caisr":
        stage_probability_labels = {
            "n1": "caisr_prob_n1",
            "n2": "caisr_prob_n2",
            "n3": "caisr_prob_n3",
            "rem": "caisr_prob_r",
            "wake": "caisr_prob_w",
        }

        stage_probability_values = []
        stage_probability_valid_ratios = []

        complete_stage_probabilities = all(
            label in signals
            for label
            in stage_probability_labels.values()
        )

        if complete_stage_probabilities:
            for stage_name, label in (
                stage_probability_labels.items()
            ):
                mean_probability, valid_ratio = (
                    aggregate_probability(
                        signals[label],
                        rates[label],
                        n_epochs,
                    )
                )

                output[
                    f"caisr_prob_{stage_name}"
                ] = mean_probability

                stage_probability_values.append(
                    mean_probability
                )

                stage_probability_valid_ratios.append(
                    valid_ratio
                )

            probability_matrix = np.column_stack(
                stage_probability_values
            )

            validity_matrix = np.column_stack(
                stage_probability_valid_ratios
            )

            probability_valid = (
                np.all(
                    np.isfinite(
                        probability_matrix
                    ),
                    axis=1,
                )
                & np.all(
                    validity_matrix >= 0.999,
                    axis=1,
                )
            )

            probability_sum = np.sum(
                probability_matrix,
                axis=1,
            )

            sum_error = np.abs(
                probability_sum - 1.0
            )

            probability_valid &= (
                sum_error <= 0.01
            )

            output[
                "caisr_stage_probability_valid"
            ] = probability_valid

            output[
                "caisr_stage_probability_sum_error"
            ] = np.where(
                np.all(
                    np.isfinite(
                        probability_matrix
                    ),
                    axis=1,
                ),
                sum_error,
                np.nan,
            )

            output[
                "caisr_stage_probability_max"
            ] = np.where(
                probability_valid,
                np.max(
                    probability_matrix,
                    axis=1,
                ),
                np.nan,
            )

            normalized = np.where(
                probability_valid[:, None],
                probability_matrix,
                np.nan,
            )

            clipped = np.clip(
                normalized,
                1e-12,
                1.0,
            )

            entropy = -np.nansum(
                normalized * np.log(clipped),
                axis=1,
            )

            entropy /= math.log(5.0)

            entropy[~probability_valid] = np.nan

            output[
                "caisr_stage_probability_entropy"
            ] = entropy

        arousal_probability_labels = {
            "no_arousal": "caisr_prob_no-ar",
            "arousal": "caisr_prob_arous",
        }

        if all(
            label in signals
            for label
            in arousal_probability_labels.values()
        ):
            no_arousal, no_arousal_valid = (
                aggregate_probability(
                    signals[
                        arousal_probability_labels[
                            "no_arousal"
                        ]
                    ],
                    rates[
                        arousal_probability_labels[
                            "no_arousal"
                        ]
                    ],
                    n_epochs,
                )
            )

            arousal, arousal_valid = (
                aggregate_probability(
                    signals[
                        arousal_probability_labels[
                            "arousal"
                        ]
                    ],
                    rates[
                        arousal_probability_labels[
                            "arousal"
                        ]
                    ],
                    n_epochs,
                )
            )

            output[
                "caisr_prob_no_arousal"
            ] = no_arousal

            output[
                "caisr_prob_arousal"
            ] = arousal

            output[
                "caisr_arousal_probability_valid_ratio"
            ] = np.minimum(
                no_arousal_valid,
                arousal_valid,
            )

            probability_sum = (
                no_arousal + arousal
            )

            output[
                "caisr_arousal_probability_sum_error"
            ] = np.where(
                np.isfinite(probability_sum),
                np.abs(
                    probability_sum - 1.0
                ),
                np.nan,
            )

    return output, detected_labels


manifest = pd.read_csv(MANIFEST_FILE)
psg_records = pd.read_csv(PSG_RECORD_FILE)

required_manifest_columns = {
    "record_id",
    "has_caisr",
    "has_human",
    "caisr_path",
    "human_path",
}

required_psg_columns = {
    "record_id",
    "site",
    "duration_sec",
    "relative_path",
}

missing_manifest_columns = (
    required_manifest_columns
    - set(manifest.columns)
)

missing_psg_columns = (
    required_psg_columns
    - set(psg_records.columns)
)

if missing_manifest_columns:
    raise RuntimeError(
        "Missing manifest columns: "
        + ", ".join(
            sorted(missing_manifest_columns)
        )
    )

if missing_psg_columns:
    raise RuntimeError(
        "Missing PSG header columns: "
        + ", ".join(
            sorted(missing_psg_columns)
        )
    )

manifest = manifest.drop(
    columns=[
        "site",
        "duration_sec",
        "relative_path",
    ],
    errors="ignore",
)

records = manifest.merge(
    psg_records[
        [
            "record_id",
            "site",
            "duration_sec",
            "relative_path",
        ]
    ],
    on="record_id",
    how="inner",
    validate="one_to_one",
)

if len(records) != len(psg_records):
    raise RuntimeError(
        "Record count mismatch after merging "
        f"manifest and PSG headers: "
        f"{len(records)} versus {len(psg_records)}"
    )

print("=== Preflight ===")
print("Records:", len(records))
print(
    "Records with CAISR:",
    int(
        records["has_caisr"]
        .map(as_bool)
        .sum()
    ),
)
print(
    "Records with human annotations:",
    int(
        records["has_human"]
        .map(as_bool)
        .sum()
    ),
)

alignment_frames = []
error_rows = []

for record_number, record in records.sort_values(
    ["site", "record_id"]
).reset_index(drop=True).iterrows():

    record_id = str(record["record_id"])
    site = str(record["site"])

    psg_duration_sec = float(
        record["duration_sec"]
    )

    n_epochs = int(
        math.floor(
            psg_duration_sec
            / EPOCH_SEC
        )
    )

    if n_epochs <= 0:
        error_rows.append({
            "record_id": record_id,
            "site": site,
            "source": "psg",
            "path": str(
                DATA_ROOT
                / str(record["relative_path"])
            ),
            "error": (
                "No complete 30-second PSG epoch"
            ),
        })
        continue

    epoch_index = np.arange(
        n_epochs,
        dtype=int,
    )

    frame = pd.DataFrame({
        "record_id": record_id,
        "site": site,
        "epoch_index": epoch_index,
        "epoch_start_sec": (
            epoch_index * EPOCH_SEC
        ),
        "epoch_end_sec": (
            (epoch_index + 1)
            * EPOCH_SEC
        ),
        "psg_duration_sec": (
            psg_duration_sec
        ),
        "psg_complete_epoch_count": (
            n_epochs
        ),
        "psg_epoch_available": True,
    })

    for source in ["human", "caisr"]:
        has_column = f"has_{source}"
        path_column = f"{source}_path"

        annotation_path = None

        if as_bool(record[has_column]):
            raw_path = record[path_column]

            if pd.isna(raw_path):
                error_rows.append({
                    "record_id": record_id,
                    "site": site,
                    "source": source,
                    "path": "",
                    "error": (
                        "Manifest indicates availability "
                        "but path is missing"
                    ),
                })
            else:
                candidate = (
                    DATA_ROOT / str(raw_path)
                )

                if candidate.is_file():
                    annotation_path = candidate
                else:
                    error_rows.append({
                        "record_id": record_id,
                        "site": site,
                        "source": source,
                        "path": str(candidate),
                        "error": "File not found",
                    })

        try:
            source_data, _ = (
                extract_annotation_source(
                    annotation_path,
                    source,
                    n_epochs,
                )
            )

        except Exception as exc:
            error_rows.append({
                "record_id": record_id,
                "site": site,
                "source": source,
                "path": (
                    str(annotation_path)
                    if annotation_path
                    else ""
                ),
                "error": repr(exc),
            })

            source_data, _ = (
                extract_annotation_source(
                    None,
                    source,
                    n_epochs,
                )
            )

        for column, values in (
            source_data.items()
        ):
            frame[column] = values

    both_stage_valid = (
        frame["human_stage_valid"]
        & frame["caisr_stage_valid"]
    )

    frame[
        "human_caisr_stage_both_valid"
    ] = both_stage_valid

    frame[
        "human_caisr_stage_match"
    ] = np.where(
        both_stage_valid,
        frame["human_stage"]
        == frame["caisr_stage"],
        np.nan,
    )

    # These columns are reserved for the later quality-mask stage.
    frame["signal_quality_available"] = False
    frame["signal_quality_valid"] = np.nan
    frame["signal_quality_score"] = np.nan
    frame["signal_quality_mask_version"] = ""

    alignment_frames.append(frame)

    if (
        (record_number + 1) % 100 == 0
        or (record_number + 1)
        == len(records)
    ):
        print(
            f"Processed "
            f"{record_number + 1}/"
            f"{len(records)} records",
            flush=True,
        )


if not alignment_frames:
    raise RuntimeError(
        "No epoch-alignment rows were generated."
    )

alignment = pd.concat(
    alignment_frames,
    ignore_index=True,
)

error_df = pd.DataFrame(
    error_rows,
    columns=[
        "record_id",
        "site",
        "source",
        "path",
        "error",
    ],
)

error_df.to_csv(
    ERROR_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Record-level summary
# ------------------------------------------------------------
record_summary_rows = []

for (
    record_id,
    site,
), group in alignment.groupby(
    ["record_id", "site"],
    sort=False,
):
    row = {
        "record_id": record_id,
        "site": site,
        "epochs": len(group),
        "duration_hours": (
            len(group)
            * EPOCH_SEC
            / 3600.0
        ),
        "human_file_available": bool(
            group[
                "human_file_available"
            ].iloc[0]
        ),
        "caisr_file_available": bool(
            group[
                "caisr_file_available"
            ].iloc[0]
        ),
        "human_stage_valid_ratio": float(
            group[
                "human_stage_valid"
            ].mean()
        ),
        "caisr_stage_valid_ratio": float(
            group[
                "caisr_stage_valid"
            ].mean()
        ),
        "both_stage_valid_ratio": float(
            group[
                "human_caisr_stage_both_valid"
            ].mean()
        ),
        "stage_match_ratio": float(
            group[
                "human_caisr_stage_match"
            ].mean()
        ),
        "caisr_stage_probability_valid_ratio": float(
            group[
                "caisr_stage_probability_valid"
            ].mean()
        ),
    }

    for source in ["human", "caisr"]:
        for event in [
            "arousal",
            "respiratory",
            "limb",
        ]:
            row[
                f"{source}_{event}_mean_valid_ratio"
            ] = float(
                group[
                    f"{source}_{event}_valid_ratio"
                ].mean()
            )

            row[
                f"{source}_{event}_positive_epoch_ratio"
            ] = float(
                group[
                    f"{source}_{event}_any"
                ].mean()
            )

    record_summary_rows.append(row)

record_summary = pd.DataFrame(
    record_summary_rows
)

record_summary.to_csv(
    RECORD_SUMMARY_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Site-level summary
# ------------------------------------------------------------
site_summary_rows = []

for site, group in alignment.groupby("site"):
    site_summary_rows.append({
        "site": site,
        "records": group[
            "record_id"
        ].nunique(),
        "epochs": len(group),
        "hours": (
            len(group)
            * EPOCH_SEC
            / 3600.0
        ),
        "human_stage_valid_pct": (
            100.0
            * group[
                "human_stage_valid"
            ].mean()
        ),
        "caisr_stage_valid_pct": (
            100.0
            * group[
                "caisr_stage_valid"
            ].mean()
        ),
        "both_stage_valid_pct": (
            100.0
            * group[
                "human_caisr_stage_both_valid"
            ].mean()
        ),
        "stage_match_pct": (
            100.0
            * group[
                "human_caisr_stage_match"
            ].mean()
        ),
        "caisr_stage_probability_valid_pct": (
            100.0
            * group[
                "caisr_stage_probability_valid"
            ].mean()
        ),
    })

site_summary = pd.DataFrame(
    site_summary_rows
)

site_summary.to_csv(
    SITE_SUMMARY_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Event summary
# ------------------------------------------------------------
event_summary_rows = []

for site, group in alignment.groupby("site"):
    for source in ["human", "caisr"]:
        for event in [
            "arousal",
            "respiratory",
            "limb",
        ]:
            valid_ratio_column = (
                f"{source}_{event}_valid_ratio"
            )

            any_column = (
                f"{source}_{event}_any"
            )

            positive_fraction_column = (
                f"{source}_{event}"
                "_positive_fraction"
            )

            event_summary_rows.append({
                "site": site,
                "source": source,
                "event_type": event,
                "epochs": len(group),
                "mean_valid_ratio": float(
                    group[
                        valid_ratio_column
                    ].mean()
                ),
                "epochs_with_any_event_pct": (
                    100.0
                    * group[
                        any_column
                    ].mean()
                ),
                "mean_positive_fraction": float(
                    group[
                        positive_fraction_column
                    ].mean()
                ),
            })

event_summary = pd.DataFrame(
    event_summary_rows
)

event_summary.to_csv(
    EVENT_SUMMARY_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Save the full alignment table
# ------------------------------------------------------------
saved_output = None

try:
    alignment.to_parquet(
        PARQUET_OUTPUT,
        index=False,
    )

    saved_output = PARQUET_OUTPUT

except Exception as exc:
    print(
        "\nParquet write failed; "
        "falling back to compressed CSV."
    )
    print("Parquet error:", repr(exc))

    alignment.to_csv(
        CSV_OUTPUT,
        index=False,
        compression="gzip",
    )

    saved_output = CSV_OUTPUT


print("\n=== Epoch alignment completed ===")
print("Rows:", len(alignment))
print(
    "Records:",
    alignment["record_id"].nunique(),
)
print("Errors:", len(error_df))
print("Saved full table:", saved_output)

try:
    size_mb = (
        saved_output.stat().st_size
        / 1024**2
    )

    print(
        f"Full-table size: {size_mb:.1f} MB"
    )
except OSError:
    pass


print("\n=== Stage alignment by site ===")
print(
    site_summary
    .round(3)
    .to_string(index=False)
)


print("\n=== Event alignment by site ===")
print(
    event_summary[
        [
            "site",
            "source",
            "event_type",
            "mean_valid_ratio",
            "epochs_with_any_event_pct",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


print("\nSaved summaries:")
print(RECORD_SUMMARY_OUTPUT)
print(SITE_SUMMARY_OUTPUT)
print(EVENT_SUMMARY_OUTPUT)
print(ERROR_OUTPUT)
