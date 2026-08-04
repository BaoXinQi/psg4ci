from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import argparse
import math
import os
import time
import traceback

import h5py
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path.home() / "fast_data/physionet2026/official_small"
DATA_ROOT = ROOT / "data/full"
AUDIT = ROOT / "audit"
MANIFEST_DIR = ROOT / "manifests"
CACHE_ROOT = ROOT / "cache/full_v1/records"

RECORD_HEADERS_FILE = AUDIT / "psg_record_headers.csv"
CHANNEL_HEADERS_FILE = AUDIT / "psg_channel_headers.csv"
DERIVATION_PLAN_FILE = AUDIT / "canonical_derivation_plan_v3_by_record.csv"
ALIGNMENT_FILE = MANIFEST_DIR / "epoch_alignment_30s_v1.parquet"
BASE_MANIFEST_FILE = AUDIT / "subject_session_record_manifest.csv"
REPRESENTATIVE_FILE = AUDIT / "representative_records_for_waveform_review.csv"
BUILD_RESULTS_FILE = AUDIT / "full_v1_build_results.csv"
BUILD_FAILURES_FILE = AUDIT / "full_v1_build_failures.csv"
FINAL_MANIFEST_CSV = MANIFEST_DIR / "full_v1_record_manifest.csv"
FINAL_MANIFEST_PARQUET = MANIFEST_DIR / "full_v1_record_manifest.parquet"

PREPROCESSING_VERSION = "full_v1"
QUALITY_VERSION = "quality_v3_final"
EPOCH_SEC = 30
SUBWINDOW_SEC = 5
SUBWINDOWS_PER_EPOCH = EPOCH_SEC // SUBWINDOW_SEC
SIGNAL_CHUNK_EPOCHS = 5
FLOAT16_STORAGE_LIMIT = 1000.0

CANONICAL_BY_MODALITY = {
    "eeg": ["EEG_F3", "EEG_F4", "EEG_C3", "EEG_C4", "EEG_O1", "EEG_O2"],
    "eog": ["EOG_E1", "EOG_E2"],
    "emg": ["CHIN_EMG", "LEG_EMG_LEFT", "LEG_EMG_RIGHT"],
    "ecg": ["ECG"],
    "resp": ["NASAL_PRESSURE", "THORACIC_EFFORT", "ABDOMINAL_EFFORT"],
    "spo2": ["SPO2"],
}

TARGET_SAMPLING_RATES = {"eeg": 128, "eog": 128, "emg": 128, "ecg": 128, "resp": 32, "spo2": 1}
FILTER_SPECS = {"eeg": (0.3, 35.0), "eog": (0.3, 35.0), "emg": (10.0, 45.0), "ecg": (0.5, 40.0), "resp": (0.05, 5.0)}
EXTREME_ROBUST_Z = {"eeg": 7.0, "eog": 7.0, "emg": 8.0, "resp": 7.0}
ABSOLUTE_PEAK_THRESHOLDS = {"eeg": 50.0, "eog": 50.0, "emg": 100.0, "resp": 50.0}
TRANSIENT_PEAK_THRESHOLDS = {"eeg": 15.0, "eog": 15.0, "emg": 30.0, "resp": 15.0}
TRANSIENT_CREST_THRESHOLDS = {"eeg": 8.0, "eog": 8.0, "emg": 10.0, "resp": 8.0}
TRANSIENT_DIFF_PEAK_THRESHOLDS = {"eeg": 20.0, "eog": 20.0, "emg": 40.0, "resp": 20.0}
GLOBAL_DROPOUT_MODALITIES = ["eeg", "eog", "emg", "ecg", "resp"]
MULTIMODAL_EXTREME_MODALITIES = ["eeg", "eog", "emg", "resp"]

G_RECORDS = None
G_CHANNELS = None
G_PLAN = None
G_ALIGNMENT = None


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def decode_scalar(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def split_source_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [label.strip() for label in str(value).split("|") if label.strip()]


def write_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def pad_or_trim(values: np.ndarray, target_length: int, fill_value: float = np.nan) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size >= target_length:
        return values[:target_length]
    output = np.full(target_length, fill_value, dtype=float)
    output[: values.size] = values
    return output


def interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    indices = np.arange(values.size)
    output = values.copy()
    output[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return output


def apply_bandpass(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = interpolate_nonfinite(values)
    nyquist = sampling_rate / 2.0
    effective_high = min(high_hz, nyquist * 0.90)
    if effective_high <= low_hz:
        raise ValueError(f"Invalid passband {low_hz:g}-{effective_high:g} Hz at {sampling_rate:g} Hz")
    sos = butter(4, [low_hz / nyquist, effective_high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, values)


def resample_signal(values: np.ndarray, source_rate: float, target_rate: int, target_length: int) -> np.ndarray:
    rounded_rate = int(round(source_rate))
    if not np.isclose(source_rate, rounded_rate, atol=1e-6):
        raise ValueError(f"Non-integer-like source sampling rate: {source_rate}")
    ratio = Fraction(target_rate, rounded_rate)
    return pad_or_trim(resample_poly(values, up=ratio.numerator, down=ratio.denominator), target_length)


def reshape_epoch_signal(values: np.ndarray, n_epochs: int, sampling_rate: int) -> np.ndarray:
    samples_per_epoch = EPOCH_SEC * sampling_rate
    return pad_or_trim(values, n_epochs * samples_per_epoch).reshape(n_epochs, samples_per_epoch)


def epoch_signal_to_5s_windows(epoch_signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    epoch_signal = np.asarray(epoch_signal, dtype=np.float32)
    if epoch_signal.ndim != 2 or epoch_signal.shape[1] != EPOCH_SEC * sampling_rate:
        raise ValueError(f"Invalid epoch signal shape {epoch_signal.shape} at {sampling_rate} Hz")
    return epoch_signal.reshape(epoch_signal.shape[0], SUBWINDOWS_PER_EPOCH, SUBWINDOW_SEC * sampling_rate).reshape(-1, SUBWINDOW_SEC * sampling_rate)


def raw_signal_to_5s_windows(values: np.ndarray, sampling_rate: float, n_epochs: int) -> np.ndarray:
    samples_float = SUBWINDOW_SEC * sampling_rate
    samples = int(round(samples_float))
    if not np.isclose(samples_float, samples, atol=1e-5):
        raise ValueError(f"Sampling rate does not map to 5-second windows: {sampling_rate}")
    n_windows = n_epochs * SUBWINDOWS_PER_EPOCH
    return pad_or_trim(values, n_windows * samples).reshape(n_windows, samples)


def calculate_window_metrics(windows: np.ndarray) -> dict[str, np.ndarray]:
    windows = np.asarray(windows, dtype=np.float32)
    finite = np.isfinite(windows)
    finite_count = finite.sum(axis=-1)
    sample_count = windows.shape[-1]
    finite_ratio = finite_count / sample_count
    safe = np.where(finite, windows, 0.0)
    mean = np.divide(safe.sum(axis=-1), finite_count, out=np.zeros_like(finite_count, dtype=np.float32), where=finite_count > 0)
    centered = np.where(finite, windows - mean[..., None], 0.0)
    variance = np.divide(np.square(centered).sum(axis=-1), finite_count, out=np.zeros_like(finite_count, dtype=np.float32), where=finite_count > 0)
    standard_deviation = np.sqrt(variance)
    rms = np.sqrt(np.divide(np.square(safe).sum(axis=-1), finite_count, out=np.zeros_like(finite_count, dtype=np.float32), where=finite_count > 0))
    peak = np.max(np.abs(safe), axis=-1)
    zero_count = (finite & np.isclose(windows, 0.0, atol=1e-8)).sum(axis=-1)
    zero_ratio = np.divide(zero_count, finite_count, out=np.ones_like(finite_count, dtype=np.float32), where=finite_count > 0)

    if sample_count > 1:
        left = windows[..., :-1]
        right = windows[..., 1:]
        valid_pairs = np.isfinite(left) & np.isfinite(right)
        valid_pair_count = valid_pairs.sum(axis=-1)
        equal_pair_count = (valid_pairs & (left == right)).sum(axis=-1)
        flat_difference_ratio = np.divide(equal_pair_count, valid_pair_count, out=np.ones_like(valid_pair_count, dtype=np.float32), where=valid_pair_count > 0)
        difference = np.where(valid_pairs, right - left, 0.0)
        difference_rms = np.sqrt(np.divide(np.square(difference).sum(axis=-1), valid_pair_count, out=np.zeros_like(valid_pair_count, dtype=np.float32), where=valid_pair_count > 0))
        difference_peak = np.max(np.abs(difference), axis=-1)
    else:
        flat_difference_ratio = np.ones_like(finite_count, dtype=np.float32)
        difference_rms = np.zeros_like(finite_count, dtype=np.float32)
        difference_peak = np.zeros_like(finite_count, dtype=np.float32)

    crest_factor = np.divide(peak, rms, out=np.zeros_like(peak, dtype=np.float32), where=rms > 1e-8)
    return {
        "finite_ratio": finite_ratio,
        "standard_deviation": standard_deviation,
        "zero_ratio": zero_ratio,
        "flat_difference_ratio": flat_difference_ratio,
        "rms": rms,
        "peak": peak,
        "difference_rms": difference_rms,
        "difference_peak": difference_peak,
        "crest_factor": crest_factor,
    }


def calculate_hard_valid_5s(raw_windows: np.ndarray) -> np.ndarray:
    metrics = calculate_window_metrics(raw_windows)
    return (
        (metrics["finite_ratio"] >= 0.99)
        & (metrics["standard_deviation"] > 1e-12)
        & (metrics["zero_ratio"] < 0.995)
        & (metrics["flat_difference_ratio"] < 0.9995)
    )


def robust_center_scale_from_5s(epoch_signal: np.ndarray, hard_valid_5s: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, float, float, float]:
    windows = epoch_signal_to_5s_windows(epoch_signal, sampling_rate)
    if windows.shape[0] != hard_valid_5s.size:
        raise ValueError("Hard-valid mask length does not match signal windows")
    valid_values = windows[hard_valid_5s].reshape(-1)
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        return np.zeros_like(epoch_signal, dtype=np.float32), 0.0, 1.0, 0.0
    center = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(valid_values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    normalized = (epoch_signal - center) / scale
    clipping_fraction = float((np.isfinite(normalized) & (np.abs(normalized) > FLOAT16_STORAGE_LIMIT)).mean())
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=FLOAT16_STORAGE_LIMIT, neginf=-FLOAT16_STORAGE_LIMIT)
    normalized = np.clip(normalized, -FLOAT16_STORAGE_LIMIT, FLOAT16_STORAGE_LIMIT).astype(np.float32)
    return normalized, center, scale, clipping_fraction


def robust_upper_outlier(values: np.ndarray, valid: np.ndarray, robust_z: float) -> np.ndarray:
    output = np.zeros(values.shape, dtype=bool)
    selected = valid & np.isfinite(values)
    selected_values = values[selected]
    if selected_values.size < 20:
        return output
    transformed = np.log1p(np.maximum(selected_values, 0.0))
    median = float(np.median(transformed))
    mad = float(np.median(np.abs(transformed - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(transformed))
    if not np.isfinite(scale) or scale <= 1e-8:
        return output
    threshold = median + robust_z * scale
    transformed_all = np.log1p(np.maximum(values, 0.0))
    output[selected] = transformed_all[selected] > threshold
    return output


def calculate_extreme_activity_5s(normalized_epoch_signal: np.ndarray, hard_valid_5s: np.ndarray, modality: str, sampling_rate: int) -> np.ndarray:
    if modality == "ecg":
        return np.zeros_like(hard_valid_5s, dtype=bool)
    windows = epoch_signal_to_5s_windows(normalized_epoch_signal, sampling_rate)
    metrics = calculate_window_metrics(windows)
    robust_z = EXTREME_ROBUST_Z[modality]
    extreme = (
        robust_upper_outlier(metrics["rms"], hard_valid_5s, robust_z)
        | robust_upper_outlier(metrics["peak"], hard_valid_5s, robust_z)
        | robust_upper_outlier(metrics["difference_rms"], hard_valid_5s, robust_z)
        | robust_upper_outlier(metrics["difference_peak"], hard_valid_5s, robust_z)
    )
    extreme |= metrics["peak"] > ABSOLUTE_PEAK_THRESHOLDS[modality]
    extreme |= (metrics["peak"] > TRANSIENT_PEAK_THRESHOLDS[modality]) & (metrics["crest_factor"] > TRANSIENT_CREST_THRESHOLDS[modality])
    extreme |= metrics["difference_peak"] > TRANSIENT_DIFF_PEAK_THRESHOLDS[modality]
    return extreme & hard_valid_5s


def aggregate_5s_to_30s(values: np.ndarray, n_epochs: int) -> np.ndarray:
    expected = n_epochs * SUBWINDOWS_PER_EPOCH
    if values.shape[0] != expected:
        raise ValueError(f"5-second mask length mismatch: {values.shape[0]} versus {expected}")
    return values.reshape(n_epochs, SUBWINDOWS_PER_EPOCH, *values.shape[1:]).mean(axis=1)


def process_spo2(raw_values: np.ndarray, sampling_rate: float, n_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]:
    rounded_rate = int(round(sampling_rate))
    if not np.isclose(sampling_rate, rounded_rate, atol=1e-6):
        raise ValueError(f"SpO2 sampling rate is not integer-like: {sampling_rate}")
    n_seconds = n_epochs * EPOCH_SEC
    values = pad_or_trim(raw_values, n_seconds * rounded_rate)
    second_matrix = values.reshape(n_seconds, rounded_rate)
    with np.errstate(all="ignore"):
        second_values = np.nanmedian(second_matrix, axis=1)
    positive_finite = second_values[np.isfinite(second_values) & (second_values > 0.0)]
    scale_indicator = float(np.median(positive_finite)) if positive_finite.size else np.nan
    if np.isfinite(scale_indicator) and scale_indicator <= 1.5:
        spo2_percent = second_values * 100.0
        scale_type = "fraction_0_to_1"
    else:
        spo2_percent = second_values
        scale_type = "percent_0_to_100"
    second_valid = np.isfinite(spo2_percent) & (spo2_percent >= 50.0) & (spo2_percent <= 100.5)
    stored = np.where(second_valid, spo2_percent, np.nan).reshape(n_epochs, EPOCH_SEC).astype(np.float32)
    hard_5s = second_valid.reshape(-1, SUBWINDOW_SEC).mean(axis=1) >= 0.80
    extreme_5s = np.zeros(hard_5s.shape, dtype=bool)
    for i, row in enumerate(stored.reshape(-1, SUBWINDOW_SEC)):
        valid_values = row[np.isfinite(row)]
        if valid_values.size < 2:
            continue
        extreme_5s[i] = ((np.max(valid_values) - np.min(valid_values)) > 10.0) or (np.max(np.abs(np.diff(valid_values))) > 5.0)
    extreme_5s &= hard_5s
    return stored, hard_5s, extreme_5s, second_valid.reshape(n_epochs, EPOCH_SEC), scale_type, scale_indicator


def write_annotation_series(group: h5py.Group, name: str, series: pd.Series) -> None:
    if (name == "stage" or name.endswith("_stage")) and "prob" not in name:
        values = pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int8).to_numpy()
        group.create_dataset(name, data=values)
        return
    if (
        pd.api.types.is_bool_dtype(series.dtype)
        or name.endswith("_valid")
        or name.endswith("_available")
        or name.endswith("_both_valid")
        or name.endswith("_match")
    ):
        values = series.fillna(False).map(as_bool).to_numpy(dtype=bool)
        group.create_dataset(name, data=values, compression="lzf")
        return
    numeric = pd.to_numeric(series, errors="coerce")
    group.create_dataset(
        name,
        data=numeric.to_numpy(dtype=np.float32),
    )


def write_annotations(parent_group: h5py.Group, alignment: pd.DataFrame) -> None:
    timing = parent_group.create_group("timing")
    timing.create_dataset("epoch_index", data=alignment["epoch_index"].to_numpy(dtype=np.int32))
    timing.create_dataset("epoch_start_sec", data=alignment["epoch_start_sec"].to_numpy(dtype=np.float32))
    timing.create_dataset("epoch_end_sec", data=alignment["epoch_end_sec"].to_numpy(dtype=np.float32))
    for group_name, prefix in {"human": "human_", "caisr": "caisr_", "agreement": "human_caisr_"}.items():
        columns = [column for column in alignment.columns if column.startswith(prefix)]
        if not columns:
            continue
        group = parent_group.create_group(group_name)
        for column in columns:
            write_annotation_series(group, column[len(prefix):], alignment[column])


def initialize_worker() -> None:
    global G_RECORDS, G_CHANNELS, G_PLAN, G_ALIGNMENT
    G_RECORDS = pd.read_csv(RECORD_HEADERS_FILE).set_index("record_id", drop=False).sort_index()
    G_CHANNELS = pd.read_csv(CHANNEL_HEADERS_FILE).set_index("record_id", drop=False).sort_index()
    plan = pd.read_csv(DERIVATION_PLAN_FILE)
    plan["available_after_standardization"] = plan["available_after_standardization"].map(as_bool)
    plan["derived_from_components"] = plan["derived_from_components"].map(as_bool)
    G_PLAN = plan.set_index(["record_id", "target_group"], drop=False).sort_index()
    G_ALIGNMENT = pd.read_parquet(ALIGNMENT_FILE).set_index("record_id", drop=False).sort_index()


def get_record_rows(frame: pd.DataFrame, record_id: str) -> pd.DataFrame:
    rows = frame.loc[[record_id]]
    return rows.to_frame().T if isinstance(rows, pd.Series) else rows.copy()


def clean_stale_temp_files(record_id: str) -> None:
    for path in CACHE_ROOT.glob(f"{record_id}.tmp.*.h5"):
        try:
            path.unlink()
        except OSError:
            pass


def validate_cache_file(path: Path, expected_record_id: str | None = None) -> tuple[bool, str]:
    if not path.is_file():
        return False, "File not found"
    try:
        with h5py.File(path, "r") as handle:
            record_id = decode_scalar(handle.attrs["record_id"])
            if expected_record_id is not None and record_id != expected_record_id:
                return False, "Record ID mismatch"
            if decode_scalar(handle.attrs["preprocessing_version"]) != PREPROCESSING_VERSION:
                return False, "Preprocessing version mismatch"
            n_epochs = int(handle.attrs["complete_epoch_count"])
            if n_epochs <= 0:
                return False, "Non-positive epoch count"
            n_windows = n_epochs * SUBWINDOWS_PER_EPOCH
            for modality, channel_names in CANONICAL_BY_MODALITY.items():
                expected_signal_shape = (n_epochs, len(channel_names), EPOCH_SEC * TARGET_SAMPLING_RATES[modality])
                signal_path = f"signals/{modality}"
                if signal_path not in handle or handle[signal_path].shape != expected_signal_shape:
                    return False, f"Signal shape mismatch for {modality}"
                expected_mask_shape = (n_windows, len(channel_names))
                for base in ["channel_hard_valid_5s", "channel_extreme_activity_5s"]:
                    mask_path = f"quality/{base}/{modality}"
                    if mask_path not in handle or handle[mask_path].shape != expected_mask_shape:
                        return False, f"Mask shape mismatch for {mask_path}"
            required = [
                "annotations/timing/epoch_index",
                "quality/modality_hard_valid_5s",
                "quality/modality_extreme_activity_5s",
                "quality/global_dropout_5s",
                "quality/multimodal_extreme_count_5s",
            ]
            for item in required:
                if item not in handle:
                    return False, f"Missing {item}"
            if handle["annotations/timing/epoch_index"].shape[0] != n_epochs:
                return False, "Annotation epoch count mismatch"
            if handle["quality/global_dropout_5s"].shape[0] != n_windows:
                return False, "Global dropout length mismatch"

            string_annotation_paths = []

            def collect_annotation_dtypes(
                name: str,
                item: h5py.Dataset | h5py.Group,
            ) -> None:
                if (
                    isinstance(item, h5py.Dataset)
                    and item.dtype.kind in {"S", "U", "O"}
                ):
                    string_annotation_paths.append(
                        f"annotations/{name}"
                    )

            handle["annotations"].visititems(
                collect_annotation_dtypes
            )

            if string_annotation_paths:
                return (
                    False,
                    "String annotation datasets found: "
                    + " | ".join(
                        string_annotation_paths[:10]
                    ),
                )

        return True, "ok"
    except Exception as exc:
        return False, repr(exc)



def summarize_existing_cache(path: Path, started: float) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        record_id = decode_scalar(handle.attrs["record_id"])
        site = decode_scalar(handle.attrs["site"])
        n_epochs = int(handle.attrs["complete_epoch_count"])
        result = {
            "record_id": record_id,
            "site": site,
            "status": "skipped_valid",
            "epochs": n_epochs,
            "duration_hours": n_epochs * EPOCH_SEC / 3600.0,
            "global_dropout_5s_ratio": float(np.asarray(handle["quality/global_dropout_5s"], dtype=bool).mean()),
            "multimodal_extreme_activity_5s_ratio": float(np.asarray(handle["quality/multimodal_extreme_activity_5s"], dtype=bool).mean()),
            "spo2_scale_type": decode_scalar(handle.attrs.get("spo2_scale_type", "")),
            "cache_path": str(path),
            "cache_size_mb": path.stat().st_size / 1024**2,
            "processing_time_sec": time.perf_counter() - started,
            "error": "",
        }
        for modality in CANONICAL_BY_MODALITY:
            present = np.asarray(handle[f"quality/channel_present/{modality}"], dtype=bool)
            hard = np.asarray(handle[f"quality/channel_hard_valid_5s/{modality}"], dtype=bool)
            result[f"{modality}_channels_present"] = int(present.sum())
            result[f"{modality}_mean_channel_hard_valid_5s_ratio"] = float(hard[:, present].mean()) if present.any() else np.nan
        return result

def process_record(record_id: str, overwrite: bool) -> dict[str, object]:
    global G_RECORDS, G_CHANNELS, G_PLAN, G_ALIGNMENT
    if any(item is None for item in [G_RECORDS, G_CHANNELS, G_PLAN, G_ALIGNMENT]):
        raise RuntimeError("Worker data were not initialized")

    started = time.perf_counter()
    final_path = CACHE_ROOT / f"{record_id}.h5"
    clean_stale_temp_files(record_id)

    if final_path.is_file() and not overwrite:
        valid, _ = validate_cache_file(final_path, expected_record_id=record_id)
        if valid:
            return summarize_existing_cache(final_path, started)
        final_path.unlink()

    temp_path = CACHE_ROOT / f"{record_id}.tmp.{os.getpid()}.h5"
    temp_path.unlink(missing_ok=True)

    record = G_RECORDS.loc[record_id]
    if isinstance(record, pd.DataFrame):
        if len(record) != 1:
            raise RuntimeError(f"Duplicate record header rows for {record_id}")
        record = record.iloc[0]

    site = str(record["site"])
    duration_sec = float(record["duration_sec"])
    n_epochs = int(math.floor(duration_sec / EPOCH_SEC))
    if n_epochs <= 0:
        raise RuntimeError("No complete 30-second epochs")

    record_channels = get_record_rows(G_CHANNELS, record_id)
    record_plan = G_PLAN.loc[record_id]
    if isinstance(record_plan, pd.Series):
        record_plan = record_plan.to_frame().T
    record_plan = record_plan.set_index("target_group", drop=False)
    alignment = get_record_rows(G_ALIGNMENT, record_id).sort_values("epoch_index").reset_index(drop=True)
    if len(alignment) != n_epochs:
        raise RuntimeError(f"Alignment row count mismatch: {len(alignment)} versus {n_epochs}")

    header_lookup = {str(row.channel_label): row for row in record_channels.itertuples(index=False)}
    psg_path = DATA_ROOT / str(record["relative_path"])
    if not psg_path.is_file():
        raise FileNotFoundError(f"PSG file not found: {psg_path}")

    signals = {}
    channel_present = {}
    channel_hard_valid_5s = {}
    channel_hard_fraction_30s = {}
    channel_extreme_5s = {}
    channel_extreme_fraction_30s = {}
    centers = {}
    scales = {}
    clipped_fractions = {}
    native_rates = {}
    methods = {}
    references = {}
    source_labels_by_modality = {}

    spo2_second_valid = np.zeros((n_epochs, EPOCH_SEC), dtype=bool)
    spo2_scale_type = ""
    spo2_scale_indicator = np.nan
    raw_signal_cache = {}

    reader = pyedflib.EdfReader(str(psg_path))
    try:
        for modality, canonical_names in CANONICAL_BY_MODALITY.items():
            target_rate = TARGET_SAMPLING_RATES[modality]
            samples_per_epoch = EPOCH_SEC * target_rate
            n_windows = n_epochs * SUBWINDOWS_PER_EPOCH

            modality_signals = np.zeros((n_epochs, len(canonical_names), samples_per_epoch), dtype=np.float32)
            modality_present = np.zeros(len(canonical_names), dtype=bool)
            modality_hard = np.zeros((n_windows, len(canonical_names)), dtype=bool)
            modality_extreme = np.zeros((n_windows, len(canonical_names)), dtype=bool)
            modality_centers = np.zeros(len(canonical_names), dtype=np.float32)
            modality_scales = np.ones(len(canonical_names), dtype=np.float32)
            modality_clipped = np.zeros(len(canonical_names), dtype=np.float32)
            modality_native_rates = np.full(len(canonical_names), np.nan, dtype=np.float32)
            modality_methods = []
            modality_references = []
            modality_source_labels = []

            for channel_index, canonical_name in enumerate(canonical_names):
                if canonical_name not in record_plan.index:
                    raise RuntimeError(f"Missing derivation-plan row for {record_id}/{canonical_name}")
                plan_row = record_plan.loc[canonical_name]
                if isinstance(plan_row, pd.DataFrame):
                    if len(plan_row) != 1:
                        raise RuntimeError(f"Duplicate derivation-plan rows for {record_id}/{canonical_name}")
                    plan_row = plan_row.iloc[0]

                method = str(plan_row["method"])
                reference = "" if pd.isna(plan_row["reference_system"]) else str(plan_row["reference_system"])
                source_labels = split_source_labels(plan_row["source_labels"])
                modality_methods.append(method)
                modality_references.append(reference)
                modality_source_labels.append(" | ".join(source_labels))

                if not as_bool(plan_row["available_after_standardization"]):
                    continue
                if not source_labels:
                    raise RuntimeError(f"Available channel has no source labels: {canonical_name}")

                source_signals = []
                source_rates = []
                for source_label in source_labels:
                    header = header_lookup.get(source_label)
                    if header is None:
                        raise RuntimeError(f"Source label not found in headers: {source_label}")
                    source_rate = float(header.sampling_rate_hz)
                    source_rates.append(source_rate)
                    if source_label not in raw_signal_cache:
                        raw_signal_cache[source_label] = np.asarray(reader.readSignal(int(header.channel_index)), dtype=float)
                    source_signals.append(raw_signal_cache[source_label])

                unique_rates = sorted(set(source_rates))
                if len(unique_rates) != 1:
                    raise RuntimeError(f"Source rates do not match for {canonical_name}: {unique_rates}")
                native_rate = unique_rates[0]

                if len(source_signals) == 1:
                    canonical_raw = source_signals[0].astype(float, copy=True)
                elif len(source_signals) == 2:
                    common_length = min(source_signals[0].size, source_signals[1].size)
                    canonical_raw = source_signals[0][:common_length] - source_signals[1][:common_length]
                else:
                    raise RuntimeError(f"Unsupported source count for {canonical_name}: {len(source_signals)}")

                canonical_raw = pad_or_trim(canonical_raw, int(round(n_epochs * EPOCH_SEC * native_rate)))
                modality_present[channel_index] = True
                modality_native_rates[channel_index] = native_rate

                if modality == "spo2":
                    spo2_epochs, hard_5s, extreme_5s, second_valid, spo2_scale_type, spo2_scale_indicator = process_spo2(
                        canonical_raw, native_rate, n_epochs
                    )
                    modality_signals[:, channel_index, :] = spo2_epochs
                    modality_hard[:, channel_index] = hard_5s
                    modality_extreme[:, channel_index] = extreme_5s
                    spo2_second_valid = second_valid
                    continue

                hard_5s = calculate_hard_valid_5s(raw_signal_to_5s_windows(canonical_raw, native_rate, n_epochs))
                low_hz, high_hz = FILTER_SPECS[modality]
                filtered = apply_bandpass(canonical_raw, native_rate, low_hz, high_hz)
                target_length = n_epochs * EPOCH_SEC * target_rate
                resampled = resample_signal(filtered, native_rate, target_rate, target_length)
                epoch_signal = reshape_epoch_signal(resampled, n_epochs, target_rate)
                normalized, center, scale, clipped_fraction = robust_center_scale_from_5s(epoch_signal, hard_5s, target_rate)
                extreme_5s = calculate_extreme_activity_5s(normalized, hard_5s, modality, target_rate)

                modality_signals[:, channel_index, :] = normalized
                modality_hard[:, channel_index] = hard_5s
                modality_extreme[:, channel_index] = extreme_5s
                modality_centers[channel_index] = center
                modality_scales[channel_index] = scale
                modality_clipped[channel_index] = clipped_fraction

            signals[modality] = modality_signals.astype(np.float16)
            channel_present[modality] = modality_present
            channel_hard_valid_5s[modality] = modality_hard
            channel_hard_fraction_30s[modality] = aggregate_5s_to_30s(modality_hard.astype(np.float32), n_epochs)
            channel_extreme_5s[modality] = modality_extreme
            channel_extreme_fraction_30s[modality] = aggregate_5s_to_30s(modality_extreme.astype(np.float32), n_epochs)
            centers[modality] = modality_centers
            scales[modality] = modality_scales
            clipped_fractions[modality] = modality_clipped
            native_rates[modality] = modality_native_rates
            methods[modality] = modality_methods
            references[modality] = modality_references
            source_labels_by_modality[modality] = modality_source_labels
    finally:
        reader.close()

    modality_names = list(CANONICAL_BY_MODALITY.keys())
    n_windows = n_epochs * SUBWINDOWS_PER_EPOCH
    modality_hard_valid_5s = np.zeros((n_windows, len(modality_names)), dtype=bool)
    modality_extreme_activity_5s = np.zeros((n_windows, len(modality_names)), dtype=bool)
    modality_available = np.zeros(len(modality_names), dtype=bool)

    for modality_index, modality in enumerate(modality_names):
        present = channel_present[modality]
        present_count = int(present.sum())
        modality_available[modality_index] = present_count > 0
        if present_count == 0:
            continue
        hard = channel_hard_valid_5s[modality][:, present]
        extreme = channel_extreme_5s[modality][:, present]
        modality_hard_valid_5s[:, modality_index] = hard.mean(axis=1) >= 0.50
        required_extreme = max(1, int(math.ceil(0.33 * present_count)))
        modality_extreme_activity_5s[:, modality_index] = extreme.sum(axis=1) >= required_extreme
        modality_extreme_activity_5s[:, modality_index] &= modality_hard_valid_5s[:, modality_index]

    modality_hard_fraction_30s = aggregate_5s_to_30s(modality_hard_valid_5s.astype(np.float32), n_epochs)
    modality_extreme_fraction_30s = aggregate_5s_to_30s(modality_extreme_activity_5s.astype(np.float32), n_epochs)
    modality_index = {name: index for index, name in enumerate(modality_names)}

    dropout_indices = [
        modality_index[name]
        for name in GLOBAL_DROPOUT_MODALITIES
        if name in modality_index and modality_available[modality_index[name]]
    ]
    if len(dropout_indices) >= 3:
        invalid_count = (~modality_hard_valid_5s[:, dropout_indices]).sum(axis=1)
        required_invalid = max(3, int(math.ceil(0.60 * len(dropout_indices))))
        global_dropout_5s = invalid_count >= required_invalid
    else:
        global_dropout_5s = np.zeros(n_windows, dtype=bool)
    global_dropout_fraction_30s = aggregate_5s_to_30s(global_dropout_5s.astype(np.float32), n_epochs)

    extreme_indices = [
        modality_index[name]
        for name in MULTIMODAL_EXTREME_MODALITIES
        if name in modality_index and modality_available[modality_index[name]]
    ]
    if extreme_indices:
        multimodal_extreme_count_5s = modality_extreme_activity_5s[:, extreme_indices].sum(axis=1).astype(np.uint8)
    else:
        multimodal_extreme_count_5s = np.zeros(n_windows, dtype=np.uint8)
    multimodal_extreme_activity_5s = multimodal_extreme_count_5s >= 2
    multimodal_extreme_fraction_30s = aggregate_5s_to_30s(multimodal_extreme_activity_5s.astype(np.float32), n_epochs)

    chunk_epochs = min(SIGNAL_CHUNK_EPOCHS, n_epochs)
    with h5py.File(temp_path, "w") as handle:
        handle.attrs["record_id"] = record_id
        handle.attrs["site"] = site
        handle.attrs["preprocessing_version"] = PREPROCESSING_VERSION
        handle.attrs["quality_version"] = QUALITY_VERSION
        handle.attrs["complete_epoch_count"] = n_epochs
        handle.attrs["epoch_duration_sec"] = EPOCH_SEC
        handle.attrs["quality_subwindow_sec"] = SUBWINDOW_SEC
        handle.attrs["source_psg_duration_sec"] = duration_sec
        handle.attrs["storage_dtype"] = "float16"
        handle.attrs["waveform_compression"] = "none"
        handle.attrs["signal_chunk_epochs"] = chunk_epochs
        handle.attrs["float16_storage_clip_abs"] = FLOAT16_STORAGE_LIMIT
        handle.attrs["spo2_scale_type"] = spo2_scale_type
        handle.attrs["spo2_scale_indicator"] = spo2_scale_indicator if np.isfinite(spo2_scale_indicator) else np.nan

        signal_group = handle.create_group("signals")
        metadata_group = handle.create_group("metadata")
        normalization_group = handle.create_group("normalization")
        quality_group = handle.create_group("quality")
        present_group = quality_group.create_group("channel_present")
        hard_group = quality_group.create_group("channel_hard_valid_5s")
        hard_fraction_group = quality_group.create_group("channel_hard_valid_fraction_30s")
        extreme_group = quality_group.create_group("channel_extreme_activity_5s")
        extreme_fraction_group = quality_group.create_group("channel_extreme_activity_fraction_30s")
        write_string_dataset(metadata_group, "modality_names", modality_names)

        for modality in modality_names:
            data = signals[modality]
            signal_group.create_dataset(
                modality,
                data=data,
                dtype=np.float16,
                chunks=(chunk_epochs, data.shape[1], data.shape[2]),
                compression=None,
            )
            present_group.create_dataset(modality, data=channel_present[modality])
            hard_group.create_dataset(modality, data=channel_hard_valid_5s[modality], compression="lzf")
            hard_fraction_group.create_dataset(modality, data=channel_hard_fraction_30s[modality].astype(np.float32), compression="lzf")
            extreme_group.create_dataset(modality, data=channel_extreme_5s[modality], compression="lzf")
            extreme_fraction_group.create_dataset(modality, data=channel_extreme_fraction_30s[modality].astype(np.float32), compression="lzf")

            normalization_group.create_dataset(f"{modality}_center", data=centers[modality].astype(np.float32))
            normalization_group.create_dataset(f"{modality}_scale", data=scales[modality].astype(np.float32))
            normalization_group.create_dataset(f"{modality}_storage_clipped_fraction", data=clipped_fractions[modality].astype(np.float32))

            modality_metadata = metadata_group.create_group(modality)
            modality_metadata.attrs["target_sampling_rate_hz"] = TARGET_SAMPLING_RATES[modality]
            if modality in FILTER_SPECS:
                modality_metadata.attrs["bandpass_low_hz"] = FILTER_SPECS[modality][0]
                modality_metadata.attrs["bandpass_high_hz"] = FILTER_SPECS[modality][1]
            write_string_dataset(modality_metadata, "channel_names", CANONICAL_BY_MODALITY[modality])
            write_string_dataset(modality_metadata, "source_labels", source_labels_by_modality[modality])
            write_string_dataset(modality_metadata, "derivation_methods", methods[modality])
            write_string_dataset(modality_metadata, "reference_systems", references[modality])
            modality_metadata.create_dataset("native_sampling_rate_hz", data=native_rates[modality].astype(np.float32))

        quality_group.create_dataset("modality_available", data=modality_available)
        quality_group.create_dataset("modality_hard_valid_5s", data=modality_hard_valid_5s, compression="lzf")
        quality_group.create_dataset("modality_hard_valid_fraction_30s", data=modality_hard_fraction_30s.astype(np.float32), compression="lzf")
        quality_group.create_dataset("modality_extreme_activity_5s", data=modality_extreme_activity_5s, compression="lzf")
        quality_group.create_dataset("modality_extreme_activity_fraction_30s", data=modality_extreme_fraction_30s.astype(np.float32), compression="lzf")
        quality_group.create_dataset("global_dropout_5s", data=global_dropout_5s, compression="lzf")
        quality_group.create_dataset("global_dropout_fraction_30s", data=global_dropout_fraction_30s.astype(np.float32), compression="lzf")
        quality_group.create_dataset("multimodal_extreme_count_5s", data=multimodal_extreme_count_5s, compression="lzf")
        quality_group.create_dataset("multimodal_extreme_activity_5s", data=multimodal_extreme_activity_5s, compression="lzf")
        quality_group.create_dataset("multimodal_extreme_activity_fraction_30s", data=multimodal_extreme_fraction_30s.astype(np.float32), compression="lzf")
        quality_group.create_dataset("spo2_second_valid", data=spo2_second_valid, compression="lzf")

        annotation_group = handle.create_group("annotations")
        write_annotations(annotation_group, alignment)

    valid, message = validate_cache_file(temp_path, expected_record_id=record_id)
    if not valid:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Temporary cache validation failed: " + message)
    temp_path.replace(final_path)

    result = {
        "record_id": record_id,
        "site": site,
        "status": "built",
        "epochs": n_epochs,
        "duration_hours": n_epochs * EPOCH_SEC / 3600.0,
        "global_dropout_5s_ratio": float(global_dropout_5s.mean()),
        "multimodal_extreme_activity_5s_ratio": float(multimodal_extreme_activity_5s.mean()),
        "spo2_scale_type": spo2_scale_type,
        "cache_path": str(final_path),
        "cache_size_mb": final_path.stat().st_size / 1024**2,
        "processing_time_sec": time.perf_counter() - started,
        "error": "",
    }
    for modality in modality_names:
        present = channel_present[modality]
        result[f"{modality}_channels_present"] = int(present.sum())
        result[f"{modality}_mean_channel_hard_valid_5s_ratio"] = (
            float(channel_hard_valid_5s[modality][:, present].mean()) if present.any() else np.nan
        )
    return result


def process_record_safe(record_id: str, overwrite: bool) -> dict[str, object]:
    try:
        return process_record(record_id, overwrite)
    except Exception as exc:
        return {
            "record_id": record_id,
            "status": "failed",
            "cache_path": "",
            "cache_size_mb": np.nan,
            "processing_time_sec": np.nan,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def save_progress(results: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(results)
    atomic_write_csv(frame, BUILD_RESULTS_FILE)
    if "status" in frame.columns:
        failures = frame[frame["status"].eq("failed")].copy()
    else:
        failures = pd.DataFrame()
    atomic_write_csv(failures, BUILD_FAILURES_FILE)


def build_final_manifest(results: pd.DataFrame) -> None:
    base_manifest = pd.read_csv(BASE_MANIFEST_FILE if BASE_MANIFEST_FILE.is_file() else RECORD_HEADERS_FILE)
    result_columns = [column for column in results.columns if column != "traceback"]
    merged = base_manifest.merge(
        results[result_columns],
        on="record_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_build"),
    )
    temp_csv = FINAL_MANIFEST_CSV.with_suffix(".csv.tmp")
    temp_parquet = FINAL_MANIFEST_PARQUET.with_suffix(".parquet.tmp")
    merged.to_csv(temp_csv, index=False)
    merged.to_parquet(temp_parquet, index=False)
    temp_csv.replace(FINAL_MANIFEST_CSV)
    temp_parquet.replace(FINAL_MANIFEST_PARQUET)


def read_record_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "record_id" not in frame.columns:
            raise RuntimeError("CSV record list must contain a record_id column")
        return frame["record_id"].astype(str).drop_duplicates().tolist()
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_record_ids(args: argparse.Namespace) -> list[str]:
    records = pd.read_csv(RECORD_HEADERS_FILE).sort_values(["site", "record_id"])
    available_ids = set(records["record_id"].astype(str))
    if args.representative_only:
        record_ids = pd.read_csv(REPRESENTATIVE_FILE)["record_id"].astype(str).drop_duplicates().tolist()
    elif args.record_list:
        record_ids = read_record_list(Path(args.record_list))
    else:
        record_ids = records["record_id"].astype(str).tolist()
    missing = sorted(set(record_ids) - available_ids)
    if missing:
        raise RuntimeError("Requested records are missing from the PSG table: " + ", ".join(missing[:20]))
    if args.max_records is not None:
        record_ids = record_ids[: args.max_records]
    return record_ids


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final PhysioNet 2026 modality-separated HDF5 cache.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel record workers. Default: 4.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild valid existing cache files.")
    parser.add_argument("--representative-only", action="store_true", help="Process only the eight representative records.")
    parser.add_argument("--record-list", type=str, default=None, help="Text or CSV file containing record IDs.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional maximum number of selected records.")
    parser.add_argument("--progress-every", type=int, default=10, help="Write progress CSVs every N completions.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    record_ids = resolve_record_ids(args)

    print("=== Full cache builder preflight ===")
    print("Selected records:", len(record_ids))
    print("Workers:", args.workers)
    print("Overwrite:", args.overwrite)
    print("Output directory:", CACHE_ROOT)
    print("Waveform layout: epoch-major, modality-separated")
    print("Waveform dtype: float16")
    print("Waveform compression: none")
    print("Signal chunk size:", SIGNAL_CHUNK_EPOCHS, "epochs")
    print("Quality resolution:", SUBWINDOW_SEC, "seconds")

    results = []
    started = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as executor:
        future_to_record = {
            executor.submit(process_record_safe, record_id, args.overwrite): record_id
            for record_id in record_ids
        }
        for completed, future in enumerate(as_completed(future_to_record), start=1):
            record_id = future_to_record[future]
            result = future.result()
            results.append(result)
            status = str(result["status"])
            if status == "failed":
                print(f"[{completed}/{len(record_ids)}] FAILED {record_id}: {result['error']}", flush=True)
            else:
                size_mb = float(result.get("cache_size_mb", np.nan))
                elapsed_sec = float(result.get("processing_time_sec", np.nan))
                print(f"[{completed}/{len(record_ids)}] {status.upper()} {record_id} | {size_mb:.1f} MB | {elapsed_sec:.1f} s", flush=True)
            if completed % args.progress_every == 0 or completed == len(record_ids):
                save_progress(results)

    result_frame = pd.DataFrame(results).sort_values("record_id").reset_index(drop=True)
    save_progress(result_frame.to_dict(orient="records"))
    build_final_manifest(result_frame)

    built_count = int(result_frame["status"].eq("built").sum())
    skipped_count = int(result_frame["status"].eq("skipped_valid").sum())
    failed_count = int(result_frame["status"].eq("failed").sum())
    valid_cache_count = 0
    for record_id in record_ids:
        valid, _ = validate_cache_file(CACHE_ROOT / f"{record_id}.h5", expected_record_id=record_id)
        valid_cache_count += int(valid)

    total_cache_gib = sum(path.stat().st_size for path in CACHE_ROOT.glob("*.h5")) / 1024**3
    total_elapsed_sec = time.perf_counter() - started

    print("\n=== Full cache builder completed ===")
    print("Selected records:", len(record_ids))
    print("Built:", built_count)
    print("Skipped valid:", skipped_count)
    print("Failed:", failed_count)
    print("Valid selected cache files:", valid_cache_count)
    print("All cache files in output:", len(list(CACHE_ROOT.glob("*.h5"))))
    print("Total cache size:", f"{total_cache_gib:.2f} GiB")
    print("Elapsed time:", f"{total_elapsed_sec / 3600.0:.2f} hours")
    if built_count:
        built_times = pd.to_numeric(
            result_frame.loc[result_frame["status"].eq("built"), "processing_time_sec"],
            errors="coerce",
        )
        print("Median build time per record:", f"{built_times.median():.1f} seconds")

    print("\nSaved:")
    print(BUILD_RESULTS_FILE)
    print(BUILD_FAILURES_FILE)
    print(FINAL_MANIFEST_CSV)
    print(FINAL_MANIFEST_PARQUET)
    print(CACHE_ROOT)

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
