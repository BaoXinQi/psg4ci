#!/usr/bin/env python3
"""Online EDF-to-feature path matching the frozen Large feature contract."""

from __future__ import annotations

import ast
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pyedflib

import caisr_feature_extractor
import preprocessing_primitives as pp
import psg_feature_extractor
from channel_mapper import ChannelMapper


SCRIPT_DIR = Path(__file__).resolve().parent
CHANNEL_TABLE = SCRIPT_DIR / "channel_table.csv"
ANNOTATION_ALIGNMENT_SOURCE = SCRIPT_DIR / "annotation_alignment_source.py"
EPOCH_SEC = 30
MAXIMUM_EPOCHS_PER_CHANNEL = 240
MINIMUM_VALID_FRACTION = 0.80
MAXIMUM_EXTREME_FRACTION = 0.20
DESATURATION_BASELINE_SECONDS = 120
MINIMUM_DESATURATION_DURATION_SECONDS = 10


@dataclass(frozen=True)
class SourceSpec:
    indices: tuple[int, ...]
    labels: tuple[str, ...]
    method: str
    reference: str = ""


def _first(values: Iterable[int]) -> int | None:
    return next(iter(values), None)


def _mapped_indices(mapper: ChannelMapper, labels: list[str]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        info = mapper.map_channel(label)
        if info.matched and info.canonical_name:
            output.setdefault(info.canonical_name, []).append(index)
    return output


def _direct(mapped: dict[str, list[int]], canonical: str, labels: list[str]) -> SourceSpec | None:
    index = _first(mapped.get(canonical, []))
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), "direct")


def _direct_filtered(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    rejected_names: set[str],
) -> SourceSpec | None:
    index = _first(
        candidate
        for candidate in mapped.get(canonical, [])
        if mapper.normalize_channel_name(labels[candidate]) not in rejected_names
    )
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), "direct")


def _direct_preferred(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    preferred_names: list[str],
) -> SourceSpec | None:
    candidates = mapped.get(canonical, [])
    if not candidates:
        return None
    rank = {name: index for index, name in enumerate(preferred_names)}
    selected = min(
        candidates,
        key=lambda index: (
            rank.get(mapper.normalize_channel_name(labels[index]), len(rank)),
            index,
        ),
    )
    return SourceSpec((selected,), (labels[selected],), "direct")


def _component(
    mapped: dict[str, list[int]],
    positive: str,
    negative: str,
    labels: list[str],
    mapper: ChannelMapper,
) -> SourceSpec | None:
    positive_candidates = mapped.get(positive, [])
    negative_candidates = mapped.get(negative, [])
    for positive_index in positive_candidates:
        normalized = mapper.normalize_channel_name(labels[positive_index])
        accepted = {
            positive.lower(),
            f"eeg {positive.lower()}",
            f"eog {positive.lower()}",
        }
        if normalized not in accepted:
            continue
        for negative_index in negative_candidates:
            negative_normalized = mapper.normalize_channel_name(labels[negative_index])
            negative_aliases = {
                negative.lower(),
                f"eeg {negative.lower()}",
            }
            if negative == "M1":
                negative_aliases.add("a1")
            elif negative == "M2":
                negative_aliases.add("a2")
            if negative_normalized not in negative_aliases:
                continue
            return SourceSpec(
                (positive_index, negative_index),
                (labels[positive_index], labels[negative_index]),
                "component_minus_reference",
                negative,
            )
    return None


def _paired_by_names(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    pairs: list[tuple[set[str], set[str]]],
) -> SourceSpec | None:
    candidates = mapped.get(canonical, [])
    normalized = {index: mapper.normalize_channel_name(labels[index]) for index in candidates}
    for positive_names, negative_names in pairs:
        positive = _first(index for index in candidates if normalized[index] in positive_names)
        negative = _first(index for index in candidates if normalized[index] in negative_names)
        if positive is not None and negative is not None:
            return SourceSpec(
                (positive, negative),
                (labels[positive], labels[negative]),
                "positive_minus_negative",
            )
    return None


def _paired_components(
    mapped: dict[str, list[int]],
    positive_canonical: str,
    negative_canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    positive_names: set[str],
    negative_names: set[str],
) -> SourceSpec | None:
    positive = _first(
        index
        for index in mapped.get(positive_canonical, [])
        if mapper.normalize_channel_name(labels[index]) in positive_names
    )
    negative = _first(
        index
        for index in mapped.get(negative_canonical, [])
        if mapper.normalize_channel_name(labels[index]) in negative_names
    )
    if positive is None or negative is None:
        return None
    return SourceSpec(
        (positive, negative),
        (labels[positive], labels[negative]),
        "positive_minus_negative",
    )


def build_source_specs(labels: list[str]) -> dict[str, SourceSpec | None]:
    mapper = ChannelMapper(CHANNEL_TABLE)
    mapped = _mapped_indices(mapper, labels)
    specs: dict[str, SourceSpec | None] = {}

    eeg_targets = {
        "EEG_F3": ("F3-M2", "F3", "M2"),
        "EEG_F4": ("F4-M1", "F4", "M1"),
        "EEG_C3": ("C3-M2", "C3", "M2"),
        "EEG_C4": ("C4-M1", "C4", "M1"),
        "EEG_O1": ("O1-M2", "O1", "M2"),
        "EEG_O2": ("O2-M1", "O2", "M1"),
    }
    for target, (direct_name, positive, negative) in eeg_targets.items():
        specs[target] = _direct(mapped, direct_name, labels) or _component(
            mapped, positive, negative, labels, mapper
        )

    left_component = _paired_components(
        mapped,
        "LOC",
        "M2",
        labels,
        mapper,
        {"loc", "eog1", "eog l", "eog-l", "leog"},
        {"m2", "a2", "eeg m2"},
    ) or _component(mapped, "E1", "M2", labels, mapper)
    right_component = _paired_components(
        mapped,
        "ROC",
        "M1",
        labels,
        mapper,
        {"roc", "eog2", "eog r", "eog-r", "reog"},
        {"m1", "a1", "eeg m1"},
    ) or _component(mapped, "E2", "M1", labels, mapper)
    specs["EOG_E1"] = _direct_filtered(
        mapped,
        "LOC",
        labels,
        mapper,
        {"loc", "eog1", "eog l", "eog-l", "leog"},
    ) or left_component
    specs["EOG_E2"] = _direct_filtered(
        mapped,
        "ROC",
        labels,
        mapper,
        {"roc", "eog2", "eog r", "eog-r", "reog"},
    ) or right_component

    specs["CHIN_EMG"] = _paired_components(
        mapped,
        "CHIN1",
        "CHIN2",
        labels,
        mapper,
        {"chin 1", "chin1", "chin l", "chin-l", "emg1", "l chin"},
        {"chin 2", "chin2", "chin r", "chin-r", "emg2", "r chin"},
    ) or _direct(mapped, "CHIN", labels)
    specs["LEG_EMG_LEFT"] = _direct(mapped, "LAT", labels) or _paired_components(
        mapped,
        "LLEG+",
        "LLEG-",
        labels,
        mapper,
        {"l-leg1", "lat 1", "lat-u", "lat1", "lleg plus"},
        {"l-leg2", "lat 2", "lat-l", "lat2", "lleg-"},
    )
    specs["LEG_EMG_RIGHT"] = _direct_filtered(
        mapped, "RAT", labels, mapper, {"l rat"}
    ) or _paired_components(
        mapped,
        "RLEG+",
        "RLEG-",
        labels,
        mapper,
        {"r-leg1", "rat 1", "rat-u", "rat1", "rleg plus"},
        {"r-leg2", "rat 2", "rat-l", "rat2", "rleg-"},
    )

    ecg_candidates = mapped.get("EKG", [])
    ecg_pair = _paired_by_names(
        mapped,
        "EKG",
        labels,
        mapper,
        [
            ({"ecg1", "ekg-l", "ecg l", "ecg-l"}, {"ecg2", "ekg-r", "ecg r", "ecg-r"}),
            ({"ekg1"}, {"ekg2"}),
        ],
    )
    generic_ecg = _first(
        index
        for index in ecg_candidates
        if mapper.normalize_channel_name(labels[index]) in {"ecg", "ekg", "ecg ii", "ecgii"}
    )
    if generic_ecg is not None:
        specs["ECG"] = SourceSpec((generic_ecg,), (labels[generic_ecg],), "direct")
    elif ecg_pair is not None:
        specs["ECG"] = ecg_pair
    else:
        specs["ECG"] = _direct(mapped, "EKG", labels)

    specs["NASAL_PRESSURE"] = _direct_preferred(
        mapped,
        "NASAL_PRESSURE",
        labels,
        mapper,
        ["nasal pressure", "nasal", "cannula", "ptaf", "nptaf", "nasaloral", "nasal oral"],
    ) or _direct_preferred(
        mapped,
        "AIRFLOW",
        labels,
        mapper,
        ["thermistor", "thermal", "airflow", "therm"],
    )
    specs["THORACIC_EFFORT"] = _direct(mapped, "CHEST", labels)
    specs["ABDOMINAL_EFFORT"] = _direct(mapped, "ABDOMINAL", labels)
    specs["SPO2"] = _direct(mapped, "SaO2", labels)
    return specs


def _read_signal(
    reader: pyedflib.EdfReader,
    index: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if index not in cache:
        cache[index] = np.asarray(reader.readSignal(index), dtype=float)
    return cache[index]


def _write_strings(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype("utf-8"),
    )


def build_psg_cache(psg_path: Path, output_path: Path, record_id: str, site_id: str) -> int:
    reader = pyedflib.EdfReader(str(psg_path))
    try:
        labels = [str(value) for value in reader.getSignalLabels()]
        headers = reader.getSignalHeaders()
        rates = [float(header["sample_frequency"]) for header in headers]
        duration_sec = float(reader.file_duration)
        n_epochs = int(math.floor(duration_sec / EPOCH_SEC))
        if n_epochs <= 0:
            raise ValueError(f"PSG is shorter than {EPOCH_SEC} seconds: {psg_path}")

        specs = build_source_specs(labels)
        raw_cache: dict[int, np.ndarray] = {}
        signals: dict[str, np.ndarray] = {}
        channel_present: dict[str, np.ndarray] = {}
        channel_hard_valid_5s: dict[str, np.ndarray] = {}
        channel_hard_fraction_30s: dict[str, np.ndarray] = {}
        channel_extreme_5s: dict[str, np.ndarray] = {}
        channel_extreme_fraction_30s: dict[str, np.ndarray] = {}
        centers: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        clipped_fractions: dict[str, np.ndarray] = {}
        methods: dict[str, list[str]] = {}
        references: dict[str, list[str]] = {}
        source_labels: dict[str, list[str]] = {}
        native_rates: dict[str, np.ndarray] = {}
        spo2_second_valid = np.zeros((n_epochs, EPOCH_SEC), dtype=bool)

        for modality, canonical_names in pp.CANONICAL_BY_MODALITY.items():
            target_rate = pp.TARGET_SAMPLING_RATES[modality]
            samples_per_epoch = EPOCH_SEC * target_rate
            n_windows = n_epochs * pp.SUBWINDOWS_PER_EPOCH
            modality_signals = np.zeros(
                (n_epochs, len(canonical_names), samples_per_epoch), dtype=np.float32
            )
            present = np.zeros(len(canonical_names), dtype=bool)
            hard = np.zeros((n_windows, len(canonical_names)), dtype=bool)
            extreme = np.zeros((n_windows, len(canonical_names)), dtype=bool)
            modality_centers = np.zeros(len(canonical_names), dtype=np.float32)
            modality_scales = np.ones(len(canonical_names), dtype=np.float32)
            modality_clipped = np.zeros(len(canonical_names), dtype=np.float32)
            modality_rates = np.full(len(canonical_names), np.nan, dtype=np.float32)
            modality_methods: list[str] = []
            modality_references: list[str] = []
            modality_source_labels: list[str] = []

            for channel_index, canonical_name in enumerate(canonical_names):
                spec = specs.get(canonical_name)
                modality_methods.append("unavailable" if spec is None else spec.method)
                modality_references.append("" if spec is None else spec.reference)
                modality_source_labels.append("" if spec is None else " | ".join(spec.labels))
                if spec is None:
                    continue

                source_rates = [rates[index] for index in spec.indices]
                if len(set(source_rates)) != 1:
                    continue
                native_rate = source_rates[0]
                source_arrays = [_read_signal(reader, index, raw_cache) for index in spec.indices]
                if len(source_arrays) == 1:
                    canonical_raw = source_arrays[0].astype(float, copy=True)
                elif len(source_arrays) == 2:
                    common = min(len(source_arrays[0]), len(source_arrays[1]))
                    canonical_raw = source_arrays[0][:common] - source_arrays[1][:common]
                else:
                    continue

                canonical_raw = pp.pad_or_trim(
                    canonical_raw,
                    int(round(n_epochs * EPOCH_SEC * native_rate)),
                )
                present[channel_index] = True
                modality_rates[channel_index] = native_rate

                if modality == "spo2":
                    (
                        spo2_epochs,
                        hard_5s,
                        extreme_5s,
                        second_valid,
                        _,
                        _,
                    ) = pp.process_spo2(canonical_raw, native_rate, n_epochs)
                    modality_signals[:, channel_index, :] = spo2_epochs
                    hard[:, channel_index] = hard_5s
                    extreme[:, channel_index] = extreme_5s
                    spo2_second_valid = second_valid
                    continue

                hard_5s = pp.calculate_hard_valid_5s(
                    pp.raw_signal_to_5s_windows(canonical_raw, native_rate, n_epochs)
                )
                low_hz, high_hz = pp.FILTER_SPECS[modality]
                filtered = pp.apply_bandpass(canonical_raw, native_rate, low_hz, high_hz)
                target_length = n_epochs * EPOCH_SEC * target_rate
                resampled = pp.resample_signal(filtered, native_rate, target_rate, target_length)
                epoch_signal = pp.reshape_epoch_signal(resampled, n_epochs, target_rate)
                normalized, center, scale, clipped = pp.robust_center_scale_from_5s(
                    epoch_signal, hard_5s, target_rate
                )
                extreme_5s = pp.calculate_extreme_activity_5s(
                    normalized, hard_5s, modality, target_rate
                )
                modality_signals[:, channel_index, :] = normalized
                hard[:, channel_index] = hard_5s
                extreme[:, channel_index] = extreme_5s
                modality_centers[channel_index] = center
                modality_scales[channel_index] = scale
                modality_clipped[channel_index] = clipped

            signals[modality] = modality_signals.astype(np.float16)
            channel_present[modality] = present
            channel_hard_valid_5s[modality] = hard
            channel_hard_fraction_30s[modality] = pp.aggregate_5s_to_30s(
                hard.astype(np.float32), n_epochs
            )
            channel_extreme_5s[modality] = extreme
            channel_extreme_fraction_30s[modality] = pp.aggregate_5s_to_30s(
                extreme.astype(np.float32), n_epochs
            )
            centers[modality] = modality_centers
            scales[modality] = modality_scales
            clipped_fractions[modality] = modality_clipped
            methods[modality] = modality_methods
            references[modality] = modality_references
            source_labels[modality] = modality_source_labels
            native_rates[modality] = modality_rates
    finally:
        reader.close()

    modality_names = list(pp.CANONICAL_BY_MODALITY)
    n_windows = n_epochs * pp.SUBWINDOWS_PER_EPOCH
    modality_hard = np.zeros((n_windows, len(modality_names)), dtype=bool)
    modality_extreme = np.zeros((n_windows, len(modality_names)), dtype=bool)
    modality_available = np.zeros(len(modality_names), dtype=bool)
    for modality_index, modality in enumerate(modality_names):
        present = channel_present[modality]
        present_count = int(present.sum())
        modality_available[modality_index] = present_count > 0
        if present_count == 0:
            continue
        hard = channel_hard_valid_5s[modality][:, present]
        extreme = channel_extreme_5s[modality][:, present]
        modality_hard[:, modality_index] = hard.mean(axis=1) >= 0.50
        required_extreme = max(1, int(math.ceil(0.33 * present_count)))
        modality_extreme[:, modality_index] = extreme.sum(axis=1) >= required_extreme
        modality_extreme[:, modality_index] &= modality_hard[:, modality_index]

    modality_hard_fraction = pp.aggregate_5s_to_30s(modality_hard.astype(np.float32), n_epochs)
    modality_extreme_fraction = pp.aggregate_5s_to_30s(
        modality_extreme.astype(np.float32), n_epochs
    )
    modality_index = {name: index for index, name in enumerate(modality_names)}
    dropout_indices = [
        modality_index[name]
        for name in pp.GLOBAL_DROPOUT_MODALITIES
        if modality_available[modality_index[name]]
    ]
    if len(dropout_indices) >= 3:
        invalid_count = (~modality_hard[:, dropout_indices]).sum(axis=1)
        required_invalid = max(3, int(math.ceil(0.60 * len(dropout_indices))))
        global_dropout = invalid_count >= required_invalid
    else:
        global_dropout = np.zeros(n_windows, dtype=bool)

    extreme_indices = [
        modality_index[name]
        for name in pp.MULTIMODAL_EXTREME_MODALITIES
        if modality_available[modality_index[name]]
    ]
    if extreme_indices:
        multimodal_extreme_count = modality_extreme[:, extreme_indices].sum(axis=1).astype(np.uint8)
    else:
        multimodal_extreme_count = np.zeros(n_windows, dtype=np.uint8)
    multimodal_extreme = multimodal_extreme_count >= 2

    with h5py.File(output_path, "w") as handle:
        handle.attrs["record_id"] = record_id
        handle.attrs["site"] = site_id
        handle.attrs["complete_epoch_count"] = n_epochs
        signal_group = handle.create_group("signals")
        normalization = handle.create_group("normalization")
        quality = handle.create_group("quality")
        present_group = quality.create_group("channel_present")
        hard_group = quality.create_group("channel_hard_valid_5s")
        hard_fraction_group = quality.create_group("channel_hard_valid_fraction_30s")
        extreme_group = quality.create_group("channel_extreme_activity_5s")
        extreme_fraction_group = quality.create_group("channel_extreme_activity_fraction_30s")
        metadata = handle.create_group("metadata")
        _write_strings(metadata, "modality_names", modality_names)

        for modality in modality_names:
            signal_group.create_dataset(modality, data=signals[modality], dtype=np.float16)
            present_group.create_dataset(modality, data=channel_present[modality])
            hard_group.create_dataset(modality, data=channel_hard_valid_5s[modality], compression="lzf")
            hard_fraction_group.create_dataset(
                modality, data=channel_hard_fraction_30s[modality].astype(np.float32), compression="lzf"
            )
            extreme_group.create_dataset(modality, data=channel_extreme_5s[modality], compression="lzf")
            extreme_fraction_group.create_dataset(
                modality,
                data=channel_extreme_fraction_30s[modality].astype(np.float32),
                compression="lzf",
            )
            normalization.create_dataset(f"{modality}_center", data=centers[modality])
            normalization.create_dataset(f"{modality}_scale", data=scales[modality])
            normalization.create_dataset(
                f"{modality}_storage_clipped_fraction", data=clipped_fractions[modality]
            )
            modality_metadata = metadata.create_group(modality)
            _write_strings(modality_metadata, "channel_names", pp.CANONICAL_BY_MODALITY[modality])
            _write_strings(modality_metadata, "source_labels", source_labels[modality])
            _write_strings(modality_metadata, "derivation_methods", methods[modality])
            _write_strings(modality_metadata, "reference_systems", references[modality])
            modality_metadata.create_dataset(
                "native_sampling_rate_hz", data=native_rates[modality]
            )

        quality.create_dataset("modality_available", data=modality_available)
        quality.create_dataset("modality_hard_valid_5s", data=modality_hard, compression="lzf")
        quality.create_dataset(
            "modality_hard_valid_fraction_30s", data=modality_hard_fraction, compression="lzf"
        )
        quality.create_dataset("modality_extreme_activity_5s", data=modality_extreme, compression="lzf")
        quality.create_dataset(
            "modality_extreme_activity_fraction_30s",
            data=modality_extreme_fraction,
            compression="lzf",
        )
        quality.create_dataset("global_dropout_5s", data=global_dropout, compression="lzf")
        quality.create_dataset(
            "global_dropout_fraction_30s",
            data=pp.aggregate_5s_to_30s(global_dropout.astype(np.float32), n_epochs),
            compression="lzf",
        )
        quality.create_dataset(
            "multimodal_extreme_count_5s", data=multimodal_extreme_count, compression="lzf"
        )
        quality.create_dataset(
            "multimodal_extreme_activity_5s", data=multimodal_extreme, compression="lzf"
        )
        quality.create_dataset(
            "multimodal_extreme_activity_fraction_30s",
            data=pp.aggregate_5s_to_30s(multimodal_extreme.astype(np.float32), n_epochs),
            compression="lzf",
        )
        quality.create_dataset("spo2_second_valid", data=spo2_second_valid, compression="lzf")
    return n_epochs


def add_caisr_annotations(cache_path: Path, caisr_path: Path | None, n_epochs: int) -> None:
    extractor = _load_annotation_extractor()
    arrays, labels = extractor(caisr_path, "caisr", n_epochs)
    with h5py.File(cache_path, "a") as handle:
        annotations = handle.require_group("annotations")
        group = annotations.create_group("caisr")
        for full_name, values in arrays.items():
            if not full_name.startswith("caisr_"):
                raise ValueError(f"Unexpected CAISR field: {full_name}")
            name = full_name[len("caisr_") :]
            array = np.asarray(values)
            if name == "stage":
                array = np.where(np.isfinite(array), array, -1).astype(np.int8)
            elif array.dtype == bool or name.endswith("_valid") or name.endswith("_available"):
                array = array.astype(bool)
            else:
                array = array.astype(np.float32)
            group.create_dataset(name, data=array, compression="lzf")
        group.attrs["detected_labels"] = " | ".join(labels)


def _load_annotation_extractor():
    function_names = {
        "normalize_label",
        "label_semantic",
        "find_label",
        "read_annotation_edf",
        "samples_per_epoch",
        "make_epoch_matrix",
        "aggregate_stage",
        "aggregate_binary_event",
        "aggregate_probability",
        "empty_float",
        "empty_bool",
        "extract_annotation_source",
    }
    source = ANNOTATION_ALIGNMENT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ANNOTATION_ALIGNMENT_SOURCE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names
    ]
    found = {node.name for node in selected}
    if found != function_names:
        raise RuntimeError(f"Annotation source is missing functions: {sorted(function_names - found)}")
    namespace = {
        "Path": Path,
        "math": math,
        "np": np,
        "pd": __import__("pandas"),
        "pyedflib": pyedflib,
        "EPOCH_SEC": float(EPOCH_SEC),
        "VALID_STAGE_CODES": np.asarray([1, 2, 3, 4, 5]),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(ANNOTATION_ALIGNMENT_SOURCE), "exec"), namespace)
    return namespace["extract_annotation_source"]


def read_psg_epoch_count(psg_path: Path) -> int:
    reader = pyedflib.EdfReader(str(psg_path))
    try:
        return int(math.floor(float(reader.file_duration) / EPOCH_SEC))
    finally:
        reader.close()


def extract_caisr_features(
    psg_path: Path,
    caisr_path: Path | None,
    record_id: str,
) -> dict:
    n_epochs = read_psg_epoch_count(psg_path)
    with tempfile.TemporaryDirectory(prefix="psg4ci_caisr_") as directory:
        cache_path = Path(directory) / f"{record_id}.h5"
        with h5py.File(cache_path, "w") as handle:
            handle.attrs["record_id"] = record_id
            handle.attrs["complete_epoch_count"] = n_epochs
        add_caisr_annotations(cache_path, caisr_path, n_epochs)
        return caisr_feature_extractor.extract_caisr_features_from_hdf5(
            record_id, str(cache_path)
        )


def extract_all_features(
    psg_path: Path,
    caisr_path: Path | None,
    record_id: str,
    site_id: str,
) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory(prefix="psg4ci_record_") as directory:
        cache_path = Path(directory) / f"{record_id}.h5"
        n_epochs = build_psg_cache(psg_path, cache_path, record_id, site_id)
        add_caisr_annotations(cache_path, caisr_path, n_epochs)
        psg_features, failure = psg_feature_extractor.process_record(
            str(cache_path),
            MAXIMUM_EPOCHS_PER_CHANNEL,
            MINIMUM_VALID_FRACTION,
            MAXIMUM_EXTREME_FRACTION,
            DESATURATION_BASELINE_SECONDS,
            MINIMUM_DESATURATION_DURATION_SECONDS,
        )
        if failure is not None or psg_features is None:
            raise RuntimeError(f"PSG feature extraction failed: {failure}")
        caisr_features = caisr_feature_extractor.extract_caisr_features_from_hdf5(
            record_id, str(cache_path)
        )
        if caisr_features.get("feature_status") != "ok":
            raise RuntimeError(
                f"CAISR feature extraction failed: {caisr_features.get('feature_error')}"
            )
        return psg_features, caisr_features
