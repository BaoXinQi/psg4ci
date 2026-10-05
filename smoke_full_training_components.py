"""Fast deterministic checks for the V18 full-training-only components."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from domain_robust_encoder_model import MODALITY_CHANNELS
from full_training_constants import CAISR_FEATURE_COLUMNS, E1_EPOCHS, SEQUENCE_EPOCHS
from full_training_residuals import (
    complete_labeled_frame,
    fit_caisr_rule,
    fit_date_rule,
    fit_followup_rule,
)
from train_domain_robust_encoder_full import SAMPLING_RATES, read_modality_windows
from training_features import build_source_specs


def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260818)
    rows = []
    for site_index, site in enumerate(("A", "B", "C")):
        for index in range(24):
            label = index % 2
            row = {
                "record_id": f"{site}-{index}",
                "SiteID": site,
                "label": label,
                "Age": 60.0 + (index % 6),
                "CreationTime": pd.Timestamp("2010-01-01")
                + pd.Timedelta(days=site_index * 400 + index * 7 + label * 2),
                "Last_Known_Visit_Date": pd.Timestamp("2024-01-01")
                + pd.Timedelta(days=site_index * 30 + index),
            }
            for feature_index, name in enumerate(CAISR_FEATURE_COLUMNS):
                row[name] = float(
                    0.05 * label + 0.01 * site_index + rng.normal(scale=0.2)
                    + feature_index * 1e-4
                )
            rows.append(row)
    return pd.DataFrame(rows)


def check_cache_layout(dtype: np.dtype) -> None:
    path = Path(__file__).resolve().with_name(f".smoke_{np.dtype(dtype).name}_layout.h5")
    path.unlink(missing_ok=True)
    try:
        with h5py.File(path, "w") as handle:
            handle.attrs["complete_epoch_count"] = 3
            signals = handle.create_group("signals")
            quality = handle.create_group("quality")
            present = quality.create_group("channel_present")
            hard = quality.create_group("channel_hard_valid_5s")
            for modality, channels in MODALITY_CHANNELS.items():
                samples = 30 * SAMPLING_RATES[modality]
                values = np.arange(3 * channels * samples, dtype=np.float32).reshape(
                    3, channels, samples
                )
                values = (values % 97) / 10.0
                stored = (
                    np.rint(values * 256.0).astype(np.int16)
                    if np.issubdtype(dtype, np.integer)
                    else values.astype(dtype)
                )
                signals.create_dataset(modality, data=stored)
                present.create_dataset(modality, data=np.ones(channels, dtype=bool))
                hard.create_dataset(modality, data=np.ones((18, channels), dtype=bool))
        with h5py.File(path, "r") as handle:
            values, valid = read_modality_windows(
                handle, "eeg", np.asarray([2, 0, 2], dtype=np.int64)
            )
        assert values.shape == (3, MODALITY_CHANNELS["eeg"], 30 * 128)
        assert valid.all()
        assert np.array_equal(values[0], values[2])
    finally:
        path.unlink(missing_ok=True)


def check_legacy_channel_selection() -> None:
    specs = build_source_specs(
        ["E1-AVG", "E2-AVG", "ECG-LA", "ECG-RA", "Thermistor 2", "Thermistor"]
    )
    assert specs["EOG_E1"].labels == ("E1-AVG",)
    assert specs["EOG_E2"].labels == ("E2-AVG",)
    assert specs["ECG"].labels == ("ECG-LA", "ECG-RA")
    assert specs["THERMAL_AIRFLOW"].labels == ("Thermistor",)


def main() -> None:
    assert E1_EPOCHS == 15
    assert SEQUENCE_EPOCHS == {20260806: 2, 20260807: 4, 20260808: 1}
    assert len(CAISR_FEATURE_COLUMNS) == 90
    frame = complete_labeled_frame(synthetic_frame())
    date = fit_date_rule(frame)
    assert date["date_coefficient"] >= 0.0
    caisr = fit_caisr_rule(frame)
    assert len(caisr["feature_coefficients"]) == 90
    assert np.isfinite(np.asarray(caisr["feature_coefficients"])).all()
    followup = fit_followup_rule(frame)
    assert followup["risk_coefficient"] >= 0.0
    check_cache_layout(np.dtype(np.float16))
    check_cache_layout(np.dtype(np.int16))
    check_legacy_channel_selection()
    print("V18_FULL_TRAINING_COMPONENTS_OK")


if __name__ == "__main__":
    main()
