"""Build the official-training cache, annotations, and dynamic manifest."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyedflib

import caisr_feature_extractor
import online_features
import preprocessing_primitives as pp
import training_features
from full_training_constants import CAISR_FEATURE_COLUMNS


_ANNOTATION_EXTRACTOR = None
_CACHE_SAFETY_BYTES = 20 * 1024**3


def clean_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value).strip()


def first_value(row: dict[str, Any], names: tuple[str, ...], required: bool = True) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip():
            return row[name]
    if required:
        raise KeyError(f"Missing required field; tried {names}")
    return None


def parse_binary(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return int(numeric)
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "positive"}:
        return 1
    if text in {"false", "f", "no", "n", "negative"}:
        return 0
    return None


def demographics_path(data_folder: Path) -> Path:
    candidates = (
        data_folder / "demographics.csv",
        data_folder / "training_set" / "demographics.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find demographics.csv under {data_folder}")


def data_root(data_folder: Path) -> Path:
    candidates = (data_folder, data_folder / "training_set")
    for path in candidates:
        if (path / "physiological_data").is_dir():
            return path
    raise FileNotFoundError(f"Could not find physiological_data under {data_folder}")


def normalize_session(value: Any) -> str:
    session = clean_identifier(value)
    return session[4:] if session.lower().startswith("ses-") else session


def resolve_file(root: Path, folder: str, site: str, filename: str) -> Path | None:
    direct = root / folder / site / filename
    if direct.is_file():
        return direct
    parent = root / folder
    if not parent.is_dir():
        return None
    matches = list(parent.glob(f"*/{filename}"))
    return matches[0] if len(matches) == 1 else None


def source_rows(data_folder: Path) -> tuple[pd.DataFrame, Path]:
    root = data_root(data_folder)
    source = pd.read_csv(demographics_path(data_folder))
    rows: list[dict[str, Any]] = []
    for raw in source.to_dict(orient="records"):
        patient = clean_identifier(
            first_value(raw, ("BidsFolder", "bids_folder", "patient_id", "PatientID"))
        )
        site = clean_identifier(first_value(raw, ("SiteID", "site_id", "site")))
        session = normalize_session(first_value(raw, ("SessionID", "session_id", "session")))
        label = parse_binary(
            first_value(
                raw,
                ("Cognitive_Impairment", "cognitive_impairment", "label"),
                required=False,
            )
        )
        if not patient or not site or not session or label is None:
            continue
        record_id = f"{patient}_ses-{session}"
        psg = resolve_file(root, "physiological_data", site, f"{record_id}.edf")
        if psg is None:
            raise FileNotFoundError(f"PSG not found for {record_id}")
        caisr = resolve_file(
            root,
            "algorithmic_annotations",
            site,
            f"{record_id}_caisr_annotations.edf",
        )
        human = resolve_file(
            root,
            "human_annotations",
            site,
            f"{record_id}_expert_annotations.edf",
        )
        row = dict(raw)
        row.update(
            {
                "record_id": record_id,
                "patient_id": patient,
                "SiteID": site,
                "SessionID": session,
                "label": int(label),
                "Cognitive_Impairment": int(label),
                "psg_path": str(psg),
                "caisr_path": "" if caisr is None else str(caisr),
                "human_path": "" if human is None else str(human),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No labeled PSG training records were found")
    if frame["record_id"].duplicated().any():
        duplicates = frame.loc[frame["record_id"].duplicated(), "record_id"].tolist()
        raise RuntimeError(f"Duplicate training records: {duplicates[:5]}")
    if "Age" not in frame:
        raise KeyError("Training demographics are missing Age")
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["Age"])].copy()
    return frame.sort_values(["SiteID", "record_id"]).reset_index(drop=True), root


def annotation_extractor():
    global _ANNOTATION_EXTRACTOR
    if _ANNOTATION_EXTRACTOR is None:
        _ANNOTATION_EXTRACTOR = online_features._load_annotation_extractor()
    return _ANNOTATION_EXTRACTOR


def write_annotation_group(
    handle: h5py.File, source: str, path: Path | None, n_epochs: int
) -> None:
    arrays, labels = annotation_extractor()(path, source, n_epochs)
    group = handle.require_group("annotations").create_group(source)
    for full_name, values in arrays.items():
        prefix = f"{source}_"
        if not full_name.startswith(prefix):
            raise ValueError(f"Unexpected {source} field: {full_name}")
        name = full_name[len(prefix) :]
        array = np.asarray(values)
        if name == "stage":
            array = np.where(np.isfinite(array), array, -1).astype(np.int8)
        elif array.dtype == bool or name.endswith("_valid") or name.endswith("_available"):
            array = array.astype(bool)
        else:
            array = array.astype(np.float32)
        group.create_dataset(name, data=array, compression="lzf")
    group.attrs["detected_labels"] = " | ".join(labels)


def write_annotation_cache(
    path: Path, record_id: str, human_path: Path | None, caisr_path: Path | None, n_epochs: int
) -> None:
    temporary = path.with_suffix(".tmp.h5")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["record_id"] = record_id
        handle.attrs["complete_epoch_count"] = int(n_epochs)
        write_annotation_group(handle, "human", human_path, n_epochs)
        write_annotation_group(handle, "caisr", caisr_path, n_epochs)
    temporary.replace(path)


def configuration(cache_path: Path) -> tuple[str, str]:
    channel_parts: list[str] = []
    modality_parts: list[str] = []
    with h5py.File(cache_path, "r") as handle:
        for modality in ("eeg", "eog", "ecg", "resp", "spo2", "emg"):
            present = np.asarray(handle[f"quality/channel_present/{modality}"], dtype=bool)
            channel_parts.append(f"{modality}:{''.join('1' if x else '0' for x in present)}")
            modality_parts.append(f"{modality}:{int(present.any())}")
    return "|".join(channel_parts), "|".join(modality_parts)


def process_record(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    record_id = str(payload["record_id"])
    cache_path = Path(payload["cache_path"])
    annotation_path = Path(payload["annotation_path"])
    temporary = cache_path.with_suffix(".tmp.h5")
    temporary.unlink(missing_ok=True)
    try:
        n_epochs = training_features.build_psg_cache(
            Path(payload["psg_path"]), temporary, record_id, str(payload["SiteID"])
        )
        temporary.replace(cache_path)
        human = Path(payload["human_path"]) if payload.get("human_path") else None
        caisr = Path(payload["caisr_path"]) if payload.get("caisr_path") else None
        write_annotation_cache(annotation_path, record_id, human, caisr, n_epochs)
        channel_config, modality_config = configuration(cache_path)
        try:
            features = caisr_feature_extractor.extract_caisr_features_from_hdf5(
                record_id, str(annotation_path)
            )
            if features.get("feature_status") != "ok":
                features = {}
        except Exception:
            features = {}
        return {
            "record_id": record_id,
            "status": "ok",
            "canonical_path": str(cache_path),
            "annotation_path": str(annotation_path),
            "complete_epoch_count": int(n_epochs),
            "channel_config": channel_config,
            "modality_config": modality_config,
            "cache_bytes": int(cache_path.stat().st_size),
            "elapsed_sec": time.perf_counter() - started,
            **{name: features.get(name, np.nan) for name in CAISR_FEATURE_COLUMNS},
        }
    except Exception as exception:
        temporary.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
        annotation_path.unlink(missing_ok=True)
        return {
            "record_id": record_id,
            "status": "failed",
            "error": repr(exception),
            "elapsed_sec": time.perf_counter() - started,
        }


def monitor_mask(frame: pd.DataFrame, per_site: int = 40) -> pd.Series:
    selected: set[str] = set()
    for _, site_frame in frame.groupby("SiteID", sort=True):
        pieces = []
        for _, label_frame in site_frame.groupby("label", sort=True):
            pieces.append(label_frame.sort_values("record_id").head(max(1, per_site // 2)))
        chosen = pd.concat(pieces).sort_values("record_id").head(per_site)
        selected.update(chosen["record_id"].astype(str))
    return frame["record_id"].astype(str).isin(selected)


def estimated_cache_bytes(frame: pd.DataFrame) -> int:
    samples_per_epoch = sum(
        len(channels) * 30 * pp.TARGET_SAMPLING_RATES[modality]
        for modality, channels in pp.CANONICAL_BY_MODALITY.items()
    )
    total_epochs = 0
    for path in frame["psg_path"].astype(str):
        reader = pyedflib.EdfReader(path)
        try:
            total_epochs += max(0, int(float(reader.file_duration) // 30))
        finally:
            reader.close()
    # Signal arrays dominate the cache. The measured HDF5 metadata/quality
    # overhead is below 0.2%; use 5% plus a fixed reserve for annotations,
    # embeddings, checkpoints, and filesystem variance.
    signal_bytes = total_epochs * samples_per_epoch * np.dtype(np.float16).itemsize
    return int(math.ceil(signal_bytes * 1.05)) + _CACHE_SAFETY_BYTES


def prepare_training_data(
    data_folder: Path,
    workspace: Path,
    workers: int,
    verbose: bool,
    max_records: int = 0,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    frame, root = source_rows(data_folder)
    if max_records > 0 and len(frame) > max_records:
        groups = list(frame.groupby(["SiteID", "label"], sort=True))
        per_group = max(2, int(math.ceil(max_records / len(groups))))
        frame = pd.concat(
            [group.sort_values("record_id").head(per_group) for _, group in groups],
            ignore_index=True,
        ).head(max_records)
    expected_bytes = estimated_cache_bytes(frame)
    free_bytes = shutil.disk_usage(workspace.parent).free
    if expected_bytes > free_bytes:
        raise RuntimeError(
            "Insufficient temporary disk for canonical training cache: "
            f"required_with_reserve={expected_bytes} free={free_bytes}"
        )
    cache_dir = workspace / "canonical" / "records"
    annotation_dir = workspace / "annotations" / "records"
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for row in frame.to_dict(orient="records"):
        row["cache_path"] = str(cache_dir / f"{row['record_id']}.h5")
        row["annotation_path"] = str(annotation_dir / f"{row['record_id']}.h5")
        payloads.append(row)

    started = time.time()
    results: list[dict[str, Any]] = []
    worker_count = max(1, min(int(workers), os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(process_record, payload): payload["record_id"] for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if verbose and (index == 1 or index % max(1, len(payloads) // 20) == 0):
                print(f"Preprocessing {index}/{len(payloads)}", flush=True)

    result_frame = pd.DataFrame(results)
    failures = result_frame.loc[result_frame["status"].ne("ok")].copy()
    failures.to_csv(workspace / "preprocessing_failures.csv", index=False)
    success = result_frame.loc[result_frame["status"].eq("ok")].drop(columns=["status"])
    frame = frame.merge(success, on="record_id", how="inner", validate="one_to_one")
    if len(frame) < 20 or frame["SiteID"].nunique() < 2:
        raise RuntimeError(
            f"Insufficient usable training cohort: records={len(frame)} sites={frame['SiteID'].nunique()}"
        )
    for site, site_frame in frame.groupby("SiteID"):
        if site_frame["label"].nunique() != 2:
            raise RuntimeError(f"Site {site} does not contain both labels")
    frame["training_include"] = True
    frame["evaluation_include"] = monitor_mask(frame)
    for column in ("Sex", "Race"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    manifest_path = workspace / "training_manifest.parquet"
    frame.sort_values(["SiteID", "record_id"]).to_parquet(manifest_path, index=False)
    summary = {
        "status": "complete",
        "data_root": str(root),
        "records_in_demographics": int(len(payloads)),
        "records_usable": int(len(frame)),
        "records_failed": int(len(failures)),
        "positives": int(frame["label"].sum()),
        "sites": frame["SiteID"].value_counts().sort_index().astype(int).to_dict(),
        "monitor_records": int(frame["evaluation_include"].sum()),
        "cache_bytes": int(frame["cache_bytes"].sum()),
        "estimated_cache_bytes_with_reserve": int(expected_bytes),
        "free_bytes_before_preprocessing": int(free_bytes),
        "elapsed_sec": time.time() - started,
        "workers": worker_count,
    }
    (workspace / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path, frame, summary


def clear_workspace(path: Path) -> None:
    resolved = path.resolve()
    temporary_root = Path("/tmp").resolve()
    if temporary_root not in resolved.parents:
        raise ValueError(f"Refusing to clear non-temporary workspace: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
