"""Fast deterministic checks for the V19 full-training-only components."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from domain_robust_encoder_model import MODALITY_CHANNELS
from full_training_constants import CAISR_FEATURE_COLUMNS, E1_EPOCHS, SEQUENCE_EPOCHS
from full_training_residuals import (
    complete_labeled_frame,
    fit_caisr_rule,
    fit_date_rule,
    fit_followup_rule,
)
from select_adaptive_sequence_epochs import (
    logit_mean_probabilities,
    median_epoch,
    robust_epoch_from_history,
    selection_gate,
)
from train_domain_robust_encoder_full import (
    SAMPLING_RATES,
    domain_reversal_strength,
    read_modality_windows,
    reverse_gradient,
    site_classification_loss,
)
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
    strengths = [domain_reversal_strength(epoch, 15, 2, 0.02) for epoch in range(15)]
    assert strengths[:2] == [0.0, 0.0]
    assert 0.0 < strengths[2] < strengths[-1]
    assert abs(strengths[-1] - 0.02) < 1e-12
    value = torch.tensor([1.0], requires_grad=True)
    reverse_gradient(value, 0.02).sum().backward()
    assert abs(float(value.grad.item()) + 0.02) < 1e-7
    logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
    targets = torch.tensor([0, 1])
    weights = torch.tensor([0.25, 2.0])
    expected = (
        torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        * weights[targets]
    ).mean()
    assert torch.allclose(site_classification_loss(logits, targets, weights), expected)
    history = [
        {"epoch": 1, "selection": 0.7000},
        {"epoch": 2, "selection": 0.7110},
        {"epoch": 3, "selection": 0.7125},
        {"epoch": 4, "selection": 0.7090},
    ]
    assert robust_epoch_from_history(history, 0.002) == 2
    assert median_epoch([1, 4, 6]) == 4
    blended = logit_mean_probabilities(
        [np.asarray([0.2, 0.8]), np.asarray([0.4, 0.6])]
    )
    assert np.all((blended > 0.0) & (blended < 1.0))
    baseline_metrics = {
        "I0002": {"age_conditioned_auroc": 0.70},
        "I0006": {"age_conditioned_auroc": 0.71},
        "S0001": {"age_conditioned_auroc": 0.69},
    }
    candidate_metrics = {
        "I0002": {"age_conditioned_auroc": 0.704},
        "I0006": {"age_conditioned_auroc": 0.714},
        "S0001": {"age_conditioned_auroc": 0.690},
    }
    assert selection_gate(baseline_metrics, candidate_metrics)["passed"]
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
    print("V20_FULL_TRAINING_COMPONENTS_OK")


if __name__ == "__main__":
    main()
