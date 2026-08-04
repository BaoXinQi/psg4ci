#!/usr/bin/env python3
"""
Extract interpretable whole-night physiological features from the official
PhysioNet Challenge 2026 HDF5 cache.

The extractor uses only the cache schema already present under:
    signals/*
    quality/*
    normalization/*

Version 1 deliberately does not require CAISR or Human annotations. This makes
the PSG branch independently usable when annotations are absent and provides a
clean test of whether raw physiological measurements add information beyond the
existing CAISR-summary anchor.

Feature families
----------------
EEG / EOG
    Band power, relative band power, spectral entropy, spectral slope,
    RMS, line length, and first-half versus second-half changes.

Respiration
    Low-frequency band power, dominant respiratory rate, irregularity,
    amplitude statistics, and inter-channel synchrony.

ECG
    Peak-derived heart rate, RR interval, SDNN, RMSSD, pNN50, signal
    amplitude, and spectral descriptors.

SpO2
    Distribution, time below 90/88/85%, first/second-half change,
    and simple 3%/4% desaturation burden.

EMG
    RMS, line length, high-frequency band power, spectral entropy,
    and first/second-half changes.

Quality
    Channel presence, valid fraction, extreme-activity fraction,
    normalization clipping fraction, and global dropout burden.

High-rate modalities are evaluated on evenly spaced valid 30-second epochs.
SpO2 is evaluated over the complete valid night.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

# Prevent each worker from spawning many BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


VERSION = "psg_physiological_features_v1"

MODALITY_SAMPLING_RATE = {
    "eeg": 128.0,
    "eog": 128.0,
    "emg": 128.0,
    "ecg": 128.0,
    "resp": 32.0,
    "spo2": 1.0,
}

MODALITY_BANDS = {
    "eeg": {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "sigma": (12.0, 16.0),
        "beta": (16.0, 30.0),
    },
    "eog": {
        "slow": (0.5, 2.0),
        "delta": (2.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
    },
    "emg": {
        "low": (10.0, 20.0),
        "mid": (20.0, 30.0),
        "high": (30.0, 45.0),
    },
    "resp": {
        "very_slow": (0.05, 0.10),
        "slow": (0.10, 0.20),
        "normal": (0.20, 0.35),
        "fast": (0.35, 0.50),
    },
}

COMMON_MODALITIES = ("eeg", "eog", "emg", "resp")


def parse_args() -> argparse.Namespace:
    root = Path.home() / "fast_data/physionet2026"

    parser = argparse.ArgumentParser(
        description=(
            "Extract multimodal physiological summary features from the "
            "official HDF5 PSG cache."
        )
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=(
            root
            / "official_small/cache/full_v1/records"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "official_small/cache/psg_physiological_features_v1.parquet"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--maximum-epochs-per-channel",
        type=int,
        default=240,
    )
    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--maximum-extreme-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--desaturation-baseline-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--minimum-desaturation-duration-seconds",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def safe_array(
    handle: h5py.File,
    path: str,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if path not in handle:
        return None

    value = np.asarray(handle[path])

    if dtype is not None:
        value = value.astype(
            dtype,
            copy=False,
        )

    return value


def robust_summary(
    values: Iterable[float] | np.ndarray,
    prefix: str,
) -> dict[str, float]:
    array = np.asarray(
        list(values)
        if not isinstance(values, np.ndarray)
        else values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return {
            f"{prefix}__mean": np.nan,
            f"{prefix}__median": np.nan,
            f"{prefix}__std": np.nan,
            f"{prefix}__iqr": np.nan,
            f"{prefix}__p10": np.nan,
            f"{prefix}__p90": np.nan,
        }

    q10, q25, q50, q75, q90 = np.percentile(
        array,
        [10, 25, 50, 75, 90],
    )

    return {
        f"{prefix}__mean": float(np.mean(array)),
        f"{prefix}__median": float(q50),
        f"{prefix}__std": float(np.std(array)),
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__p10": float(q10),
        f"{prefix}__p90": float(q90),
    }


def half_summary(
    values: np.ndarray,
    epoch_indices: np.ndarray,
    total_epochs: int,
    prefix: str,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    epoch_indices = np.asarray(epoch_indices, dtype=int)

    finite = np.isfinite(values)
    values = values[finite]
    epoch_indices = epoch_indices[finite]

    if values.size == 0:
        return {
            f"{prefix}__first_half_median": np.nan,
            f"{prefix}__second_half_median": np.nan,
            f"{prefix}__second_minus_first": np.nan,
        }

    midpoint = total_epochs / 2.0

    first = values[
        epoch_indices < midpoint
    ]
    second = values[
        epoch_indices >= midpoint
    ]

    first_median = (
        float(np.median(first))
        if first.size
        else np.nan
    )
    second_median = (
        float(np.median(second))
        if second.size
        else np.nan
    )

    difference = (
        second_median - first_median
        if np.isfinite(first_median)
        and np.isfinite(second_median)
        else np.nan
    )

    return {
        f"{prefix}__first_half_median": first_median,
        f"{prefix}__second_half_median": second_median,
        f"{prefix}__second_minus_first": difference,
    }


def evenly_spaced_indices(
    indices: np.ndarray,
    maximum_count: int,
) -> np.ndarray:
    indices = np.asarray(
        indices,
        dtype=int,
    )

    if indices.size <= maximum_count:
        return indices

    positions = np.linspace(
        0,
        indices.size - 1,
        maximum_count,
    )

    positions = np.unique(
        np.round(positions).astype(int)
    )

    return indices[positions]


def restore_signal(
    stored: np.ndarray,
    center: float,
    scale: float,
) -> np.ndarray:
    stored = np.asarray(
        stored,
        dtype=np.float32,
    )

    if not np.isfinite(center):
        center = 0.0

    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    return (
        stored
        * float(scale)
        + float(center)
    )


def compute_epoch_spectral_features(
    epochs: np.ndarray,
    sampling_rate: float,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    epochs = np.asarray(
        epochs,
        dtype=np.float32,
    )

    epochs = epochs - np.nanmean(
        epochs,
        axis=1,
        keepdims=True,
    )

    epochs = np.nan_to_num(
        epochs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    n_samples = epochs.shape[1]

    window = np.hanning(
        n_samples
    ).astype(np.float32)

    transformed = np.fft.rfft(
        epochs * window[None, :],
        axis=1,
    )

    power = (
        np.abs(transformed) ** 2
    ).astype(np.float64)

    frequencies = np.fft.rfftfreq(
        n_samples,
        d=1.0 / sampling_rate,
    )

    total_mask = (
        frequencies >= 0.05
    ) & (
        frequencies <= min(
            sampling_rate / 2.0,
            45.0,
        )
    )

    total_power = np.sum(
        power[:, total_mask],
        axis=1,
    ) + 1e-12

    normalized_power = (
        power[:, total_mask]
        / total_power[:, None]
    )

    entropy = -np.sum(
        normalized_power
        * np.log(
            normalized_power
            + 1e-12
        ),
        axis=1,
    )

    entropy /= math.log(
        max(
            normalized_power.shape[1],
            2,
        )
    )

    output: dict[
        str,
        np.ndarray,
    ] = {
        "rms": np.sqrt(
            np.mean(
                epochs ** 2,
                axis=1,
            )
        ),
        "line_length": np.mean(
            np.abs(
                np.diff(
                    epochs,
                    axis=1,
                )
            ),
            axis=1,
        ),
        "spectral_entropy": entropy,
    }

    positive_frequency = (
        frequencies > 0
    ) & total_mask

    log_frequency = np.log10(
        frequencies[
            positive_frequency
        ]
    )

    if log_frequency.size >= 2:
        log_power = np.log10(
            power[
                :,
                positive_frequency,
            ]
            + 1e-12
        )

        centered_frequency = (
            log_frequency
            - np.mean(
                log_frequency
            )
        )

        denominator = np.sum(
            centered_frequency
            ** 2
        )

        centered_power = (
            log_power
            - np.mean(
                log_power,
                axis=1,
                keepdims=True,
            )
        )

        output[
            "spectral_slope"
        ] = (
            np.sum(
                centered_power
                * centered_frequency[
                    None,
                    :
                ],
                axis=1,
            )
            / max(
                denominator,
                1e-12,
            )
        )
    else:
        output[
            "spectral_slope"
        ] = np.full(
            epochs.shape[0],
            np.nan,
            dtype=float,
        )

    for band_name, (
        low_frequency,
        high_frequency,
    ) in bands.items():
        mask = (
            frequencies >= low_frequency
        ) & (
            frequencies < high_frequency
        )

        band_power = np.sum(
            power[:, mask],
            axis=1,
        ) + 1e-12

        output[
            f"log_power_{band_name}"
        ] = np.log10(
            band_power
        )

        output[
            f"relative_power_{band_name}"
        ] = (
            band_power
            / total_power
        )

    return output


def get_channel_quality(
    handle: h5py.File,
    modality: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    signal_shape = handle[
        f"signals/{modality}"
    ].shape

    n_epochs = int(
        signal_shape[0]
    )
    n_channels = int(
        signal_shape[1]
    )

    present = safe_array(
        handle,
        f"quality/channel_present/{modality}",
        dtype=bool,
    )

    if present is None:
        present = np.ones(
            n_channels,
            dtype=bool,
        )

    hard_valid = safe_array(
        handle,
        (
            "quality/"
            "channel_hard_valid_fraction_30s/"
            f"{modality}"
        ),
        dtype=float,
    )

    if hard_valid is None:
        hard_valid = np.ones(
            (
                n_epochs,
                n_channels,
            ),
            dtype=float,
        )

    extreme = safe_array(
        handle,
        (
            "quality/"
            "channel_extreme_activity_fraction_30s/"
            f"{modality}"
        ),
        dtype=float,
    )

    if extreme is None:
        extreme = np.zeros(
            (
                n_epochs,
                n_channels,
            ),
            dtype=float,
        )

    return (
        present,
        hard_valid,
        extreme,
    )


def get_normalization(
    handle: h5py.File,
    modality: str,
    n_channels: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    center = safe_array(
        handle,
        f"normalization/{modality}_center",
        dtype=float,
    )

    scale = safe_array(
        handle,
        f"normalization/{modality}_scale",
        dtype=float,
    )

    clipped = safe_array(
        handle,
        (
            "normalization/"
            f"{modality}_storage_clipped_fraction"
        ),
        dtype=float,
    )

    if center is None:
        center = np.zeros(
            n_channels,
            dtype=float,
        )

    if scale is None:
        scale = np.ones(
            n_channels,
            dtype=float,
        )

    if clipped is None:
        clipped = np.full(
            n_channels,
            np.nan,
            dtype=float,
        )

    return (
        np.asarray(center),
        np.asarray(scale),
        np.asarray(clipped),
    )


def extract_common_modality(
    handle: h5py.File,
    modality: str,
    maximum_epochs: int,
    minimum_valid_fraction: float,
    maximum_extreme_fraction: float,
) -> dict[str, float]:
    features: dict[
        str,
        float,
    ] = {}

    signal_dataset = handle[
        f"signals/{modality}"
    ]

    n_epochs = int(
        signal_dataset.shape[0]
    )
    n_channels = int(
        signal_dataset.shape[1]
    )

    sampling_rate = (
        MODALITY_SAMPLING_RATE[
            modality
        ]
    )

    bands = MODALITY_BANDS[
        modality
    ]

    (
        present,
        hard_valid,
        extreme,
    ) = get_channel_quality(
        handle,
        modality,
    )

    (
        center,
        scale,
        clipped,
    ) = get_normalization(
        handle,
        modality,
        n_channels,
    )

    for channel_index in range(
        n_channels
    ):
        prefix = (
            f"psg_{modality}_ch"
            f"{channel_index}"
        )

        features[
            f"{prefix}__present"
        ] = float(
            present[channel_index]
        )

        features[
            f"{prefix}__hard_valid_fraction"
        ] = float(
            np.nanmean(
                hard_valid[
                    :,
                    channel_index,
                ]
            )
        )

        features[
            f"{prefix}__extreme_activity_fraction"
        ] = float(
            np.nanmean(
                extreme[
                    :,
                    channel_index,
                ]
            )
        )

        features[
            f"{prefix}__storage_clipped_fraction"
        ] = float(
            clipped[
                channel_index
            ]
        )

        good = (
            bool(
                present[
                    channel_index
                ]
            )
            & (
                hard_valid[
                    :,
                    channel_index,
                ]
                >= minimum_valid_fraction
            )
            & (
                extreme[
                    :,
                    channel_index,
                ]
                <= maximum_extreme_fraction
            )
        )

        valid_indices = np.flatnonzero(
            good
        )

        features[
            f"{prefix}__eligible_epoch_fraction"
        ] = float(
            len(valid_indices)
            / max(
                n_epochs,
                1,
            )
        )

        selected_indices = evenly_spaced_indices(
            valid_indices,
            maximum_epochs,
        )

        features[
            f"{prefix}__sampled_epoch_count"
        ] = float(
            len(
                selected_indices
            )
        )

        if len(selected_indices) == 0:
            continue

        stored_epochs = np.asarray(
            signal_dataset[
                selected_indices,
                channel_index,
                :,
            ],
            dtype=np.float32,
        )

        epochs = restore_signal(
            stored_epochs,
            center[
                channel_index
            ],
            scale[
                channel_index
            ],
        )

        spectral = (
            compute_epoch_spectral_features(
                epochs=epochs,
                sampling_rate=(
                    sampling_rate
                ),
                bands=bands,
            )
        )

        for measure_name, values in (
            spectral.items()
        ):
            measure_prefix = (
                f"{prefix}__"
                f"{measure_name}"
            )

            features.update(
                robust_summary(
                    values,
                    measure_prefix,
                )
            )

            features.update(
                half_summary(
                    values=values,
                    epoch_indices=(
                        selected_indices
                    ),
                    total_epochs=n_epochs,
                    prefix=measure_prefix,
                )
            )

        if modality == "resp":
            respiratory_mask = (
                (
                    np.fft.rfftfreq(
                        epochs.shape[1],
                        d=(
                            1.0
                            / sampling_rate
                        ),
                    )
                    >= 0.08
                )
                & (
                    np.fft.rfftfreq(
                        epochs.shape[1],
                        d=(
                            1.0
                            / sampling_rate
                        ),
                    )
                    <= 0.60
                )
            )

            centered = (
                epochs
                - np.mean(
                    epochs,
                    axis=1,
                    keepdims=True,
                )
            )

            respiratory_power = (
                np.abs(
                    np.fft.rfft(
                        centered
                        * np.hanning(
                            epochs.shape[1]
                        )[None, :],
                        axis=1,
                    )
                )
                ** 2
            )

            respiratory_frequencies = (
                np.fft.rfftfreq(
                    epochs.shape[1],
                    d=(
                        1.0
                        / sampling_rate
                    ),
                )
            )

            masked_power = (
                respiratory_power[
                    :,
                    respiratory_mask,
                ]
            )

            dominant_index = np.argmax(
                masked_power,
                axis=1,
            )

            dominant_rate = (
                respiratory_frequencies[
                    respiratory_mask
                ][dominant_index]
                * 60.0
            )

            features.update(
                robust_summary(
                    dominant_rate,
                    (
                        f"{prefix}__"
                        "dominant_rate_bpm"
                    ),
                )
            )

            features.update(
                half_summary(
                    values=dominant_rate,
                    epoch_indices=(
                        selected_indices
                    ),
                    total_epochs=n_epochs,
                    prefix=(
                        f"{prefix}__"
                        "dominant_rate_bpm"
                    ),
                )
            )

    if modality == "resp" and n_channels >= 2:
        for first_channel in range(
            n_channels
        ):
            for second_channel in range(
                first_channel + 1,
                n_channels,
            ):
                pair_prefix = (
                    "psg_resp_sync_"
                    f"ch{first_channel}_"
                    f"ch{second_channel}"
                )

                shared_good = (
                    bool(
                        present[
                            first_channel
                        ]
                    )
                    & bool(
                        present[
                            second_channel
                        ]
                    )
                    & (
                        hard_valid[
                            :,
                            first_channel,
                        ]
                        >= minimum_valid_fraction
                    )
                    & (
                        hard_valid[
                            :,
                            second_channel,
                        ]
                        >= minimum_valid_fraction
                    )
                    & (
                        extreme[
                            :,
                            first_channel,
                        ]
                        <= maximum_extreme_fraction
                    )
                    & (
                        extreme[
                            :,
                            second_channel,
                        ]
                        <= maximum_extreme_fraction
                    )
                )

                shared_indices = (
                    evenly_spaced_indices(
                        np.flatnonzero(
                            shared_good
                        ),
                        min(
                            maximum_epochs,
                            120,
                        ),
                    )
                )

                correlations: list[
                    float
                ] = []

                if len(shared_indices):
                    first = restore_signal(
                        np.asarray(
                            signal_dataset[
                                shared_indices,
                                first_channel,
                                :,
                            ],
                            dtype=np.float32,
                        ),
                        center[
                            first_channel
                        ],
                        scale[
                            first_channel
                        ],
                    )

                    second = restore_signal(
                        np.asarray(
                            signal_dataset[
                                shared_indices,
                                second_channel,
                                :,
                            ],
                            dtype=np.float32,
                        ),
                        center[
                            second_channel
                        ],
                        scale[
                            second_channel
                        ],
                    )

                    first = (
                        first
                        - np.mean(
                            first,
                            axis=1,
                            keepdims=True,
                        )
                    )

                    second = (
                        second
                        - np.mean(
                            second,
                            axis=1,
                            keepdims=True,
                        )
                    )

                    numerator = np.sum(
                        first
                        * second,
                        axis=1,
                    )

                    denominator = np.sqrt(
                        np.sum(
                            first ** 2,
                            axis=1,
                        )
                        * np.sum(
                            second ** 2,
                            axis=1,
                        )
                    )

                    valid = (
                        denominator
                        > 1e-12
                    )

                    correlations = (
                        numerator[valid]
                        / denominator[valid]
                    )

                features.update(
                    robust_summary(
                        np.asarray(
                            correlations,
                            dtype=float,
                        ),
                        (
                            f"{pair_prefix}__"
                            "correlation"
                        ),
                    )
                )

    return features


def extract_ecg(
    handle: h5py.File,
    maximum_epochs: int,
    minimum_valid_fraction: float,
    maximum_extreme_fraction: float,
) -> dict[str, float]:
    modality = "ecg"
    features: dict[
        str,
        float,
    ] = {}

    dataset = handle[
        "signals/ecg"
    ]

    n_epochs = int(
        dataset.shape[0]
    )
    n_channels = int(
        dataset.shape[1]
    )
    sampling_rate = (
        MODALITY_SAMPLING_RATE[
            modality
        ]
    )

    (
        present,
        hard_valid,
        extreme,
    ) = get_channel_quality(
        handle,
        modality,
    )

    (
        center,
        scale,
        clipped,
    ) = get_normalization(
        handle,
        modality,
        n_channels,
    )

    for channel_index in range(
        n_channels
    ):
        prefix = (
            f"psg_ecg_ch"
            f"{channel_index}"
        )

        features[
            f"{prefix}__present"
        ] = float(
            present[
                channel_index
            ]
        )

        features[
            f"{prefix}__hard_valid_fraction"
        ] = float(
            np.nanmean(
                hard_valid[
                    :,
                    channel_index,
                ]
            )
        )

        features[
            f"{prefix}__extreme_activity_fraction"
        ] = float(
            np.nanmean(
                extreme[
                    :,
                    channel_index,
                ]
            )
        )

        features[
            f"{prefix}__storage_clipped_fraction"
        ] = float(
            clipped[
                channel_index
            ]
        )

        good = (
            bool(
                present[
                    channel_index
                ]
            )
            & (
                hard_valid[
                    :,
                    channel_index,
                ]
                >= minimum_valid_fraction
            )
            & (
                extreme[
                    :,
                    channel_index,
                ]
                <= maximum_extreme_fraction
            )
        )

        selected_indices = (
            evenly_spaced_indices(
                np.flatnonzero(
                    good
                ),
                maximum_epochs,
            )
        )

        features[
            f"{prefix}__eligible_epoch_fraction"
        ] = float(
            np.sum(
                good
            )
            / max(
                n_epochs,
                1,
            )
        )

        features[
            f"{prefix}__sampled_epoch_count"
        ] = float(
            len(
                selected_indices
            )
        )

        if len(selected_indices) == 0:
            continue

        epochs = restore_signal(
            np.asarray(
                dataset[
                    selected_indices,
                    channel_index,
                    :,
                ],
                dtype=np.float32,
            ),
            center[
                channel_index
            ],
            scale[
                channel_index
            ],
        )

        generic = (
            compute_epoch_spectral_features(
                epochs=epochs,
                sampling_rate=(
                    sampling_rate
                ),
                bands={
                    "low": (
                        0.5,
                        5.0,
                    ),
                    "mid": (
                        5.0,
                        15.0,
                    ),
                    "high": (
                        15.0,
                        40.0,
                    ),
                },
            )
        )

        for measure_name, values in (
            generic.items()
        ):
            measure_prefix = (
                f"{prefix}__"
                f"{measure_name}"
            )

            features.update(
                robust_summary(
                    values,
                    measure_prefix,
                )
            )

            features.update(
                half_summary(
                    values,
                    selected_indices,
                    n_epochs,
                    measure_prefix,
                )
            )

        epoch_heart_rate: list[
            float
        ] = []
        epoch_rmssd: list[
            float
        ] = []
        epoch_sdnn: list[
            float
        ] = []
        epoch_pnn50: list[
            float
        ] = []
        epoch_peak_count: list[
            float
        ] = []
        epoch_positions: list[
            int
        ] = []

        for epoch_index, signal in zip(
            selected_indices,
            epochs,
        ):
            signal = np.asarray(
                signal,
                dtype=float,
            )

            signal -= np.median(
                signal
            )

            positive_tail = np.percentile(
                signal,
                99,
            )
            negative_tail = abs(
                np.percentile(
                    signal,
                    1,
                )
            )

            working = (
                signal
                if positive_tail
                >= negative_tail
                else -signal
            )

            standard_deviation = float(
                np.std(
                    working
                )
            )

            if (
                not np.isfinite(
                    standard_deviation
                )
                or standard_deviation
                <= 1e-12
            ):
                continue

            peaks, _ = find_peaks(
                working,
                distance=int(
                    0.30
                    * sampling_rate
                ),
                prominence=max(
                    0.50
                    * standard_deviation,
                    1e-8,
                ),
            )

            if len(peaks) < 3:
                continue

            rr = np.diff(
                peaks
            ) / sampling_rate

            rr = rr[
                (rr >= 0.30)
                & (rr <= 2.00)
            ]

            if rr.size < 2:
                continue

            heart_rate = (
                60.0
                / np.median(
                    rr
                )
            )

            if not (
                25.0
                <= heart_rate
                <= 220.0
            ):
                continue

            epoch_heart_rate.append(
                float(
                    heart_rate
                )
            )
            epoch_sdnn.append(
                float(
                    np.std(
                        rr,
                        ddof=1,
                    )
                    * 1000.0
                )
                if rr.size >= 2
                else np.nan
            )

            rr_difference = np.diff(
                rr
            )

            epoch_rmssd.append(
                float(
                    np.sqrt(
                        np.mean(
                            rr_difference
                            ** 2
                        )
                    )
                    * 1000.0
                )
                if rr_difference.size
                else np.nan
            )

            epoch_pnn50.append(
                float(
                    np.mean(
                        np.abs(
                            rr_difference
                        )
                        > 0.050
                    )
                )
                if rr_difference.size
                else np.nan
            )

            epoch_peak_count.append(
                float(
                    len(
                        peaks
                    )
                )
            )
            epoch_positions.append(
                int(
                    epoch_index
                )
            )

        epoch_positions_array = np.asarray(
            epoch_positions,
            dtype=int,
        )

        for measure_name, values in {
            "heart_rate_bpm": (
                epoch_heart_rate
            ),
            "sdnn_ms": (
                epoch_sdnn
            ),
            "rmssd_ms": (
                epoch_rmssd
            ),
            "pnn50": (
                epoch_pnn50
            ),
            "peak_count_30s": (
                epoch_peak_count
            ),
        }.items():
            values_array = np.asarray(
                values,
                dtype=float,
            )

            measure_prefix = (
                f"{prefix}__"
                f"{measure_name}"
            )

            features.update(
                robust_summary(
                    values_array,
                    measure_prefix,
                )
            )

            features.update(
                half_summary(
                    values_array,
                    epoch_positions_array,
                    n_epochs,
                    measure_prefix,
                )
            )

        features[
            f"{prefix}__peak_valid_epoch_fraction"
        ] = float(
            len(
                epoch_heart_rate
            )
            / max(
                len(
                    selected_indices
                ),
                1,
            )
        )

    return features


def trailing_maximum(
    values: np.ndarray,
    valid: np.ndarray,
    window: int,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )
    valid = np.asarray(
        valid,
        dtype=bool,
    )

    output = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    queue: deque[
        tuple[int, float]
    ] = deque()

    for index, (
        value,
        is_valid,
    ) in enumerate(
        zip(
            values,
            valid,
        )
    ):
        if not is_valid:
            queue.clear()
            continue

        minimum_index = (
            index
            - window
            + 1
        )

        while (
            queue
            and queue[0][0]
            < minimum_index
        ):
            queue.popleft()

        while (
            queue
            and queue[-1][1]
            <= value
        ):
            queue.pop()

        queue.append(
            (
                index,
                float(
                    value
                ),
            )
        )

        output[index] = (
            queue[0][1]
        )

    return output


def desaturation_features(
    spo2: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    baseline_seconds: int,
    minimum_duration_seconds: int,
    valid_hours: float,
    prefix: str,
) -> dict[str, float]:
    baseline = trailing_maximum(
        values=spo2,
        valid=valid,
        window=baseline_seconds,
    )

    drop = (
        baseline
        - spo2
    )

    active = (
        valid
        & np.isfinite(
            drop
        )
        & (
            drop
            >= threshold
        )
    )

    event_depths: list[
        float
    ] = []
    event_durations: list[
        float
    ] = []
    event_areas: list[
        float
    ] = []

    start: int | None = None

    for index in range(
        len(
            active
        )
        + 1
    ):
        is_active = (
            bool(
                active[index]
            )
            if index < len(active)
            else False
        )

        if is_active and start is None:
            start = index

        if (
            not is_active
            and start is not None
        ):
            stop = index
            duration = stop - start

            if (
                duration
                >= minimum_duration_seconds
            ):
                segment = drop[
                    start:stop
                ]

                event_depths.append(
                    float(
                        np.nanmax(
                            segment
                        )
                    )
                )
                event_durations.append(
                    float(
                        duration
                    )
                )
                event_areas.append(
                    float(
                        np.nansum(
                            segment
                        )
                    )
                )

            start = None

    event_count = len(
        event_depths
    )

    return {
        f"{prefix}__event_count": float(
            event_count
        ),
        f"{prefix}__events_per_hour": (
            float(
                event_count
                / valid_hours
            )
            if valid_hours > 0
            else np.nan
        ),
        f"{prefix}__depth_median": (
            float(
                np.median(
                    event_depths
                )
            )
            if event_depths
            else 0.0
        ),
        f"{prefix}__depth_max": (
            float(
                np.max(
                    event_depths
                )
            )
            if event_depths
            else 0.0
        ),
        f"{prefix}__duration_median_seconds": (
            float(
                np.median(
                    event_durations
                )
            )
            if event_durations
            else 0.0
        ),
        f"{prefix}__area_per_hour": (
            float(
                np.sum(
                    event_areas
                )
                / valid_hours
            )
            if valid_hours > 0
            else np.nan
        ),
    }


def extract_spo2(
    handle: h5py.File,
    minimum_valid_fraction: float,
    maximum_extreme_fraction: float,
    baseline_seconds: int,
    minimum_duration_seconds: int,
) -> dict[str, float]:
    features: dict[
        str,
        float,
    ] = {}

    dataset = handle[
        "signals/spo2"
    ]

    n_epochs = int(
        dataset.shape[0]
    )
    n_channels = int(
        dataset.shape[1]
    )

    (
        present,
        hard_valid,
        extreme,
    ) = get_channel_quality(
        handle,
        "spo2",
    )

    (
        center,
        scale,
        clipped,
    ) = get_normalization(
        handle,
        "spo2",
        n_channels,
    )

    second_valid = safe_array(
        handle,
        "quality/spo2_second_valid",
        dtype=bool,
    )

    if second_valid is None:
        second_valid = np.ones(
            (
                n_epochs,
                30,
            ),
            dtype=bool,
        )

    for channel_index in range(
        n_channels
    ):
        prefix = (
            f"psg_spo2_ch"
            f"{channel_index}"
        )

        features[
            f"{prefix}__present"
        ] = float(
            present[
                channel_index
            ]
        )

        features[
            f"{prefix}__hard_valid_fraction"
        ] = float(
            np.nanmean(
                hard_valid[
                    :,
                    channel_index,
                ]
            )
        )

        features[
            f"{prefix}__extreme_activity_fraction"
        ] = float(
            np.nanmean(
                extreme[
                    :,
                    channel_index,
                ]
            )
        )

        features[
            f"{prefix}__storage_clipped_fraction"
        ] = float(
            clipped[
                channel_index
            ]
        )

        signal = restore_signal(
            np.asarray(
                dataset[
                    :,
                    channel_index,
                    :,
                ],
                dtype=np.float32,
            ),
            center[
                channel_index
            ],
            scale[
                channel_index
            ],
        ).reshape(-1)

        epoch_good = (
            bool(
                present[
                    channel_index
                ]
            )
            & (
                hard_valid[
                    :,
                    channel_index,
                ]
                >= minimum_valid_fraction
            )
            & (
                extreme[
                    :,
                    channel_index,
                ]
                <= maximum_extreme_fraction
            )
        )

        valid = (
            second_valid.reshape(-1)
            & np.repeat(
                epoch_good,
                30,
            )
            & np.isfinite(
                signal
            )
            & (
                signal >= 40.0
            )
            & (
                signal <= 100.5
            )
        )

        valid_values = signal[
            valid
        ]

        features[
            f"{prefix}__valid_second_fraction"
        ] = float(
            np.mean(
                valid
            )
        )

        features[
            f"{prefix}__valid_hours"
        ] = float(
            np.sum(
                valid
            )
            / 3600.0
        )

        if valid_values.size == 0:
            continue

        percentiles = np.percentile(
            valid_values,
            [
                1,
                5,
                10,
                25,
                50,
                75,
                90,
                95,
                99,
            ],
        )

        for percentile, value in zip(
            [
                1,
                5,
                10,
                25,
                50,
                75,
                90,
                95,
                99,
            ],
            percentiles,
        ):
            features[
                f"{prefix}__p{percentile:02d}"
            ] = float(
                value
            )

        features[
            f"{prefix}__mean"
        ] = float(
            np.mean(
                valid_values
            )
        )
        features[
            f"{prefix}__std"
        ] = float(
            np.std(
                valid_values
            )
        )
        features[
            f"{prefix}__minimum"
        ] = float(
            np.min(
                valid_values
            )
        )

        for threshold in (
            90.0,
            88.0,
            85.0,
        ):
            features[
                (
                    f"{prefix}__"
                    f"fraction_below_{int(threshold)}"
                )
            ] = float(
                np.mean(
                    valid_values
                    < threshold
                )
            )

        midpoint = len(
            signal
        ) // 2

        first_values = signal[
            :midpoint
        ][
            valid[
                :midpoint
            ]
        ]

        second_values = signal[
            midpoint:
        ][
            valid[
                midpoint:
            ]
        ]

        first_median = (
            float(
                np.median(
                    first_values
                )
            )
            if first_values.size
            else np.nan
        )

        second_median = (
            float(
                np.median(
                    second_values
                )
            )
            if second_values.size
            else np.nan
        )

        features[
            f"{prefix}__first_half_median"
        ] = first_median
        features[
            f"{prefix}__second_half_median"
        ] = second_median
        features[
            f"{prefix}__second_minus_first"
        ] = (
            second_median
            - first_median
            if np.isfinite(
                first_median
            )
            and np.isfinite(
                second_median
            )
            else np.nan
        )

        valid_hours = float(
            np.sum(
                valid
            )
            / 3600.0
        )

        for threshold in (
            3.0,
            4.0,
        ):
            features.update(
                desaturation_features(
                    spo2=signal,
                    valid=valid,
                    threshold=threshold,
                    baseline_seconds=(
                        baseline_seconds
                    ),
                    minimum_duration_seconds=(
                        minimum_duration_seconds
                    ),
                    valid_hours=valid_hours,
                    prefix=(
                        f"{prefix}__"
                        f"desaturation_{int(threshold)}pct"
                    ),
                )
            )

    return features


def extract_global_quality(
    handle: h5py.File,
) -> dict[str, float]:
    features: dict[
        str,
        float,
    ] = {}

    n_epochs = int(
        handle[
            "signals/eeg"
        ].shape[0]
    )

    features[
        "psg_record__epoch_count"
    ] = float(
        n_epochs
    )
    features[
        "psg_record__duration_hours"
    ] = float(
        n_epochs
        * 30.0
        / 3600.0
    )

    quality_paths = {
        "global_dropout_fraction": (
            "quality/"
            "global_dropout_fraction_30s"
        ),
        "multimodal_extreme_activity_fraction": (
            "quality/"
            "multimodal_extreme_activity_fraction_30s"
        ),
        "multimodal_extreme_count": (
            "quality/"
            "multimodal_extreme_count_5s"
        ),
    }

    for feature_name, path in (
        quality_paths.items()
    ):
        values = safe_array(
            handle,
            path,
            dtype=float,
        )

        if values is None:
            continue

        features.update(
            robust_summary(
                values.reshape(-1),
                (
                    "psg_quality__"
                    f"{feature_name}"
                ),
            )
        )

    return features


def process_record(
    path_string: str,
    maximum_epochs: int,
    minimum_valid_fraction: float,
    maximum_extreme_fraction: float,
    baseline_seconds: int,
    minimum_desaturation_duration: int,
) -> tuple[
    dict[str, Any] | None,
    dict[str, str] | None,
]:
    path = Path(
        path_string
    )

    try:
        features: dict[
            str,
            Any,
        ] = {
            "record_id": path.stem,
            "cache_path": str(path),
            "feature_status": "ok",
        }

        with h5py.File(
            path,
            "r",
        ) as handle:
            required_paths = [
                "signals/eeg",
                "signals/eog",
                "signals/emg",
                "signals/ecg",
                "signals/resp",
                "signals/spo2",
            ]

            missing = [
                required
                for required in required_paths
                if required not in handle
            ]

            if missing:
                raise KeyError(
                    "Missing required HDF5 datasets: "
                    f"{missing}"
                )

            features.update(
                extract_global_quality(
                    handle
                )
            )

            for modality in COMMON_MODALITIES:
                features.update(
                    extract_common_modality(
                        handle=handle,
                        modality=modality,
                        maximum_epochs=(
                            maximum_epochs
                        ),
                        minimum_valid_fraction=(
                            minimum_valid_fraction
                        ),
                        maximum_extreme_fraction=(
                            maximum_extreme_fraction
                        ),
                    )
                )

            features.update(
                extract_ecg(
                    handle=handle,
                    maximum_epochs=(
                        maximum_epochs
                    ),
                    minimum_valid_fraction=(
                        minimum_valid_fraction
                    ),
                    maximum_extreme_fraction=(
                        maximum_extreme_fraction
                    ),
                )
            )

            features.update(
                extract_spo2(
                    handle=handle,
                    minimum_valid_fraction=(
                        minimum_valid_fraction
                    ),
                    maximum_extreme_fraction=(
                        maximum_extreme_fraction
                    ),
                    baseline_seconds=(
                        baseline_seconds
                    ),
                    minimum_duration_seconds=(
                        minimum_desaturation_duration
                    ),
                )
            )

        return (
            features,
            None,
        )

    except Exception as exception:
        failure = {
            "record_id": path.stem,
            "cache_path": str(path),
            "error": repr(
                exception
            ),
            "traceback": traceback.format_exc(),
        }

        return (
            None,
            failure,
        )


def discover_records(
    records_dir: Path,
) -> list[Path]:
    paths = sorted(
        list(
            records_dir.glob(
                "*.h5"
            )
        )
        + list(
            records_dir.glob(
                "*.hdf5"
            )
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No HDF5 records found in: {records_dir}"
        )

    return paths


def write_output(
    frame: pd.DataFrame,
    output: Path,
) -> Path:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output.suffix.lower()

    if suffix in {
        ".parquet",
        ".pq",
    }:
        try:
            frame.to_parquet(
                output,
                index=False,
            )
            return output
        except Exception as exception:
            fallback = output.with_suffix(
                ".csv"
            )

            print(
                "WARNING: Parquet writing failed; "
                f"saving CSV instead: {exception}",
                file=sys.stderr,
                flush=True,
            )

            frame.to_csv(
                fallback,
                index=False,
            )

            return fallback

    if suffix == ".csv":
        frame.to_csv(
            output,
            index=False,
        )
        return output

    raise ValueError(
        "Output must end in .parquet, .pq, or .csv"
    )


def main() -> None:
    args = parse_args()

    if (
        args.output.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output exists: {args.output}. "
            "Use --overwrite."
        )

    if args.workers < 1:
        raise ValueError(
            "--workers must be at least 1."
        )

    paths = discover_records(
        args.records_dir
    )

    print(
        "=== PSG physiological feature extraction ===",
        flush=True,
    )
    print(
        f"Records: {len(paths)}",
        flush=True,
    )
    print(
        f"Workers: {args.workers}",
        flush=True,
    )
    print(
        "Maximum epochs per channel: "
        f"{args.maximum_epochs_per_channel}",
        flush=True,
    )

    rows: list[
        dict[str, Any]
    ] = []
    failures: list[
        dict[str, str]
    ] = []

    if args.workers == 1:
        for index, path in enumerate(
            paths,
            start=1,
        ):
            row, failure = process_record(
                str(path),
                args.maximum_epochs_per_channel,
                args.minimum_valid_fraction,
                args.maximum_extreme_fraction,
                args.desaturation_baseline_seconds,
                args.minimum_desaturation_duration_seconds,
            )

            if row is not None:
                rows.append(
                    row
                )

            if failure is not None:
                failures.append(
                    failure
                )

            if (
                index == 1
                or index == len(paths)
                or index % 25 == 0
            ):
                print(
                    (
                        f"Processed {index}/{len(paths)}; "
                        f"success={len(rows)}, "
                        f"failed={len(failures)}"
                    ),
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    process_record,
                    str(path),
                    args.maximum_epochs_per_channel,
                    args.minimum_valid_fraction,
                    args.maximum_extreme_fraction,
                    args.desaturation_baseline_seconds,
                    args.minimum_desaturation_duration_seconds,
                ): path
                for path in paths
            }

            completed = 0

            for future in as_completed(
                futures
            ):
                completed += 1
                row, failure = future.result()

                if row is not None:
                    rows.append(
                        row
                    )

                if failure is not None:
                    failures.append(
                        failure
                    )

                if (
                    completed == 1
                    or completed == len(paths)
                    or completed % 25 == 0
                ):
                    print(
                        (
                            f"Processed {completed}/{len(paths)}; "
                            f"success={len(rows)}, "
                            f"failed={len(failures)}"
                        ),
                        flush=True,
                    )

    if not rows:
        raise RuntimeError(
            "No records were successfully processed."
        )

    frame = pd.DataFrame(
        rows
    ).sort_values(
        "record_id"
    ).reset_index(
        drop=True
    )

    numeric_columns = [
        column
        for column in frame.columns
        if column.startswith(
            "psg_"
        )
    ]

    numeric_columns = [
        column
        for column in numeric_columns
        if pd.to_numeric(
            frame[column],
            errors="coerce",
        ).notna().any()
    ]

    frame = frame[
        [
            "record_id",
            "cache_path",
            "feature_status",
        ]
        + sorted(
            numeric_columns
        )
    ]

    saved_output = write_output(
        frame,
        args.output,
    )

    failure_path = (
        args.output.parent
        / (
            args.output.stem
            + "_failures.csv"
        )
    )

    pd.DataFrame(
        failures,
        columns=[
            "record_id",
            "cache_path",
            "error",
            "traceback",
        ],
    ).to_csv(
        failure_path,
        index=False,
    )

    modality_counts = {}

    for modality in [
        "eeg",
        "eog",
        "emg",
        "ecg",
        "resp",
        "spo2",
        "quality",
        "record",
    ]:
        modality_counts[
            modality
        ] = int(
            sum(
                column.startswith(
                    f"psg_{modality}"
                )
                for column in (
                    frame.columns
                )
            )
        )

    metadata = {
        "version": VERSION,
        "records_directory": str(
            args.records_dir.resolve()
        ),
        "requested_output": str(
            args.output
        ),
        "saved_output": str(
            saved_output.resolve()
        ),
        "record_count": int(
            len(paths)
        ),
        "successful_record_count": int(
            len(frame)
        ),
        "failure_count": int(
            len(failures)
        ),
        "feature_count": int(
            len(
                numeric_columns
            )
        ),
        "feature_count_by_prefix": (
            modality_counts
        ),
        "maximum_epochs_per_channel": int(
            args.maximum_epochs_per_channel
        ),
        "minimum_valid_fraction": float(
            args.minimum_valid_fraction
        ),
        "maximum_extreme_fraction": float(
            args.maximum_extreme_fraction
        ),
        "desaturation_baseline_seconds": int(
            args.desaturation_baseline_seconds
        ),
        "minimum_desaturation_duration_seconds": int(
            args.minimum_desaturation_duration_seconds
        ),
        "stage_aware_features": False,
        "annotation_dependency": False,
    }

    metadata_path = (
        args.output.parent
        / (
            args.output.stem
            + "_metadata.json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== Extraction summary ===",
        flush=True,
    )
    print(
        f"Successful records: {len(frame)}",
        flush=True,
    )
    print(
        f"Failed records: {len(failures)}",
        flush=True,
    )
    print(
        f"Features: {len(numeric_columns)}",
        flush=True,
    )
    print(
        "Features by prefix: "
        f"{modality_counts}",
        flush=True,
    )
    print(
        f"Saved: {saved_output}",
        flush=True,
    )
    print(
        f"Failures: {failure_path}",
        flush=True,
    )
    print(
        f"Metadata: {metadata_path}",
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
