#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import mne
import numpy as np
import pandas as pd

from channel_mapper import MODALITY_ORDER, group_channels_for_sleepfm


# ============================================================
# Defaults
# ============================================================
TARGET_PSG_SFREQ = 128.0
TARGET_ANN_SFREQ: Optional[float] = None
READ_VERBOSE = False
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 4
FLUSH_EVERY = 5
CACHE_VERSION = "psg_ann_hdf5_v2_channel_table"


# ============================================================
# Discovery helpers
# ============================================================
def infer_train_root(data_folder: Path | str) -> Path:
    data_folder = Path(data_folder)
    candidates = [
        data_folder / "training_set",
        data_folder,
    ]
    for c in candidates:
        if (c / "physiological_data").exists():
            return c
    raise FileNotFoundError(
        f"Could not infer training root from {data_folder}; expected physiological_data under it."
    )


def infer_physiological_root(data_folder: Path | str) -> Path:
    train_root = infer_train_root(data_folder)
    physio_root = train_root / "physiological_data"
    if not physio_root.exists():
        raise FileNotFoundError(f"physiological_data not found under {train_root}")
    return physio_root


def list_physiological_edfs(root: Path) -> List[Path]:
    return sorted(root.glob("*/*.edf"))


def infer_site_from_path(edf_path: Path) -> str:
    try:
        return edf_path.parent.name
    except Exception:
        return "unknown"


def build_index_by_basename(training_root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in training_root.rglob("*.edf"):
        idx.setdefault(p.name, []).append(p)
    return idx


def find_matching_annotation_files(psg_edf: Path, basename_index: Dict[str, List[Path]]) -> Tuple[Optional[Path], Optional[Path]]:
    stem = psg_edf.stem
    caisr_name = f"{stem}_caisr_annotations.edf"
    expert_name = f"{stem}_expert_annotations.edf"

    caisr = basename_index.get(caisr_name, [None])[0]
    expert = basename_index.get(expert_name, [None])[0]
    return caisr, expert


# ============================================================
# EDF reading
# ============================================================
def read_and_optionally_resample(
    edf_path: Path,
    target_sfreq: Optional[float],
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str], float, float]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=verbose)
    original_sfreq = float(raw.info["sfreq"])
    duration_sec = float(raw.n_times / original_sfreq)

    if target_sfreq is not None and float(raw.info["sfreq"]) != float(target_sfreq):
        raw.resample(target_sfreq, npad="auto")

    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float32, copy=False)
    ch_names = list(raw.ch_names)
    return data, ch_names, sfreq, duration_sec


# ============================================================
# HDF5 writing helpers
# ============================================================
def write_string_list_dataset(group: h5py.Group, name: str, values: List[str]):
    dt = h5py.string_dtype(encoding="utf-8")
    arr = np.asarray(values, dtype=object)
    group.create_dataset(name, data=arr, dtype=dt)


def _jsonify_grouped_channels(items: List[object]) -> str:
    payload = []
    for ch in items:
        payload.append(
            {
                "raw_name": getattr(ch, "raw_name", ""),
                "normalized_name": getattr(ch, "normalized_name", ""),
                "canonical_name": getattr(ch, "canonical_name", ""),
                "family": getattr(ch, "family", ""),
                "modality": getattr(ch, "modality", ""),
                "priority": int(getattr(ch, "priority", 0)),
                "source": getattr(ch, "source", ""),
                "original_index": int(getattr(ch, "original_index", -1)),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def write_grouped_channel_metadata(group: h5py.Group, grouped_channels: List[object]) -> None:
    write_string_list_dataset(group, "channels", [str(ch.raw_name) for ch in grouped_channels])
    write_string_list_dataset(group, "normalized_channels", [str(ch.normalized_name) for ch in grouped_channels])
    write_string_list_dataset(group, "canonical_channels", [str(ch.canonical_name) for ch in grouped_channels])
    write_string_list_dataset(group, "families", [str(ch.family) for ch in grouped_channels])
    write_string_list_dataset(group, "sources", [str(ch.source) for ch in grouped_channels])
    group.create_dataset(
        "original_indices",
        data=np.asarray([int(ch.original_index) for ch in grouped_channels], dtype=np.int32),
    )
    group.create_dataset(
        "priorities",
        data=np.asarray([int(ch.priority) for ch in grouped_channels], dtype=np.int32),
    )


def write_psg_group(
    hf: h5py.File,
    psg_data: np.ndarray,
    ch_names: List[str],
    site: str,
    file_name: str,
    src_path: str,
    sfreq: float,
    duration_sec: float,
) -> Dict[str, object]:
    g = hf.create_group("psg")
    g.attrs["site"] = site
    g.attrs["file_name"] = file_name
    g.attrs["source_path"] = src_path
    g.attrs["sfreq"] = float(sfreq)
    g.attrs["duration_sec"] = float(duration_sec)

    mapping = group_channels_for_sleepfm(
        channel_names=ch_names,
        keep_unmatched=True,
        preserve_original_order=False,
    )
    grouped = mapping["grouped"]
    dropped = mapping["dropped"]
    unmatched = mapping.get("unmatched", [])

    write_string_list_dataset(g, "all_channels", [str(x) for x in ch_names])
    write_string_list_dataset(g, "unmatched_channels", [str(x.raw_name) for x in unmatched])
    write_string_list_dataset(g, "unmatched_normalized", [str(x.normalized_name) for x in unmatched])
    g.attrs["n_total_channels"] = int(len(ch_names))
    g.attrs["n_unmatched_channels"] = int(len(unmatched))
    g.attrs["channel_mapping_json"] = json.dumps(
        {
            "grouped": {mod: json.loads(_jsonify_grouped_channels(items)) for mod, items in grouped.items()},
            "dropped": {mod: json.loads(_jsonify_grouped_channels(items)) for mod, items in dropped.items()},
            "unmatched": [
                {
                    "raw_name": x.raw_name,
                    "normalized_name": x.normalized_name,
                }
                for x in unmatched
            ],
        },
        ensure_ascii=False,
    )

    summary: Dict[str, object] = {
        "psg_n_total_channels": int(len(ch_names)),
        "psg_channels_json": json.dumps(ch_names, ensure_ascii=False),
        "psg_n_unmatched_channels": int(len(unmatched)),
        "psg_unmatched_channels_json": json.dumps([x.raw_name for x in unmatched], ensure_ascii=False),
        "psg_unmatched_normalized_json": json.dumps([x.normalized_name for x in unmatched], ensure_ascii=False),
    }

    for mod in MODALITY_ORDER:
        mod_lower = mod.lower()
        mod_grouped = grouped[mod]
        mod_dropped = dropped[mod]
        mg = g.create_group(mod_lower)
        mg.attrs["modality"] = mod
        mg.attrs["n_channels"] = int(len(mod_grouped))
        mg.attrs["n_dropped_channels"] = int(len(mod_dropped))
        mg.attrs["available"] = bool(len(mod_grouped) > 0)
        mg.attrs["dropped_channels_json"] = _jsonify_grouped_channels(mod_dropped)

        if len(mod_grouped) == 0:
            write_string_list_dataset(mg, "channels", [])
            write_string_list_dataset(mg, "normalized_channels", [])
            write_string_list_dataset(mg, "canonical_channels", [])
            write_string_list_dataset(mg, "families", [])
            write_string_list_dataset(mg, "sources", [])
            mg.create_dataset("original_indices", data=np.zeros((0,), dtype=np.int32))
            mg.create_dataset("priorities", data=np.zeros((0,), dtype=np.int32))
            mg.create_dataset("data", data=np.zeros((0, 0), dtype=np.float32))
        else:
            idx = [int(ch.original_index) for ch in mod_grouped]
            arr = psg_data[idx].astype(np.float32, copy=False)
            write_grouped_channel_metadata(mg, mod_grouped)
            mg.create_dataset(
                "data",
                data=arr,
                compression=HDF5_COMPRESSION,
                compression_opts=HDF5_COMPRESSION_OPTS,
            )

        summary.update(
            {
                f"n_{mod_lower}_channels": int(len(mod_grouped)),
                f"n_{mod_lower}_dropped_channels": int(len(mod_dropped)),
                f"{mod_lower}_channels_json": json.dumps([x.raw_name for x in mod_grouped], ensure_ascii=False),
                f"{mod_lower}_canonical_channels_json": json.dumps([x.canonical_name for x in mod_grouped], ensure_ascii=False),
                f"{mod_lower}_families_json": json.dumps([x.family for x in mod_grouped], ensure_ascii=False),
                f"{mod_lower}_dropped_channels_json": _jsonify_grouped_channels(mod_dropped),
            }
        )

    return summary


def write_annotation_group(
    hf: h5py.File,
    group_name: str,
    ann_path: Optional[Path],
    target_sfreq: Optional[float],
    verbose: bool = False,
) -> Dict[str, object]:
    out = {
        f"has_{group_name}": False,
        f"{group_name}_path": "",
        f"{group_name}_sfreq": np.nan,
        f"{group_name}_duration_sec": np.nan,
        f"{group_name}_n_channels": 0,
        f"{group_name}_channels_json": "[]",
    }

    g = hf.create_group(f"annotations/{group_name}")

    if ann_path is None or not ann_path.exists():
        g.attrs["available"] = False
        return out

    data, ch_names, sfreq, duration_sec = read_and_optionally_resample(
        ann_path, target_sfreq=target_sfreq, verbose=verbose
    )

    g.attrs["available"] = True
    g.attrs["source_path"] = str(ann_path)
    g.attrs["sfreq"] = float(sfreq)
    g.attrs["duration_sec"] = float(duration_sec)
    write_string_list_dataset(g, "channels", ch_names)
    g.create_dataset(
        "data",
        data=data.astype(np.float32, copy=False),
        compression=HDF5_COMPRESSION,
        compression_opts=HDF5_COMPRESSION_OPTS,
    )

    out.update({
        f"has_{group_name}": True,
        f"{group_name}_path": str(ann_path),
        f"{group_name}_sfreq": float(sfreq),
        f"{group_name}_duration_sec": float(duration_sec),
        f"{group_name}_n_channels": int(len(ch_names)),
        f"{group_name}_channels_json": json.dumps(ch_names, ensure_ascii=False),
    })
    return out


# ============================================================
# Main per-file processing
# ============================================================
def build_out_path(out_dir: Path, site: str, file_name: str) -> Path:
    subdir = out_dir / site
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{Path(file_name).stem}.h5"


def process_one_file(
    psg_edf: Path,
    basename_index: Dict[str, List[Path]],
    out_dir: Path,
    overwrite: bool = False,
    target_psg_sfreq: float = TARGET_PSG_SFREQ,
    target_ann_sfreq: Optional[float] = TARGET_ANN_SFREQ,
    read_verbose: bool = READ_VERBOSE,
) -> Dict[str, object]:
    t0 = time.perf_counter()

    site = infer_site_from_path(psg_edf)
    file_name = psg_edf.name
    out_path = build_out_path(out_dir, site, file_name)

    row: Dict[str, object] = {
        "status": "unknown",
        "site": site,
        "file_name": file_name,
        "psg_path": str(psg_edf),
        "cache_h5_path": str(out_path),
        "elapsed_sec_total": np.nan,
        "elapsed_sec_psg_read_resample": np.nan,
        "elapsed_sec_write": np.nan,
    }

    if out_path.exists() and not overwrite:
        row["status"] = "skip_existing"
        row["elapsed_sec_total"] = 0.0
        return row

    caisr_path, expert_path = find_matching_annotation_files(psg_edf, basename_index)
    row["caisr_path_discovered"] = "" if caisr_path is None else str(caisr_path)
    row["expert_path_discovered"] = "" if expert_path is None else str(expert_path)

    try:
        t_psg0 = time.perf_counter()
        psg_data, psg_ch_names, psg_sfreq, psg_duration = read_and_optionally_resample(
            psg_edf, target_sfreq=target_psg_sfreq, verbose=read_verbose
        )
        row["elapsed_sec_psg_read_resample"] = time.perf_counter() - t_psg0

        t_w0 = time.perf_counter()
        with h5py.File(out_path, "w") as hf:
            hf.attrs["cache_version"] = CACHE_VERSION
            hf.attrs["site"] = site
            hf.attrs["file_name"] = file_name

            psg_summary = write_psg_group(
                hf=hf,
                psg_data=psg_data,
                ch_names=psg_ch_names,
                site=site,
                file_name=file_name,
                src_path=str(psg_edf),
                sfreq=psg_sfreq,
                duration_sec=psg_duration,
            )

            ann_summary_caisr = write_annotation_group(
                hf=hf,
                group_name="caisr",
                ann_path=caisr_path,
                target_sfreq=target_ann_sfreq,
                verbose=read_verbose,
            )
            ann_summary_expert = write_annotation_group(
                hf=hf,
                group_name="expert",
                ann_path=expert_path,
                target_sfreq=target_ann_sfreq,
                verbose=read_verbose,
            )

        row["elapsed_sec_write"] = time.perf_counter() - t_w0
        row["status"] = "ok"
        row.update(
            {
                "psg_sfreq": float(psg_sfreq),
                "psg_duration_sec": float(psg_duration),
            }
        )
        row.update(psg_summary)
        row.update(ann_summary_caisr)
        row.update(ann_summary_expert)

    except Exception as e:
        row["status"] = "error"
        row["error_type"] = type(e).__name__
        row["error_message"] = str(e)
        row["traceback"] = traceback.format_exc()

    row["elapsed_sec_total"] = time.perf_counter() - t0
    return row


# ============================================================
# Public entry
# ============================================================
def build_cache(
    data_folder: Path | str,
    out_dir: Path | str,
    overwrite: bool = False,
    limit_files: Optional[int] = None,
    target_psg_sfreq: float = TARGET_PSG_SFREQ,
    target_ann_sfreq: Optional[float] = TARGET_ANN_SFREQ,
    read_verbose: bool = READ_VERBOSE,
    flush_every: int = FLUSH_EVERY,
    metadata_csv: Optional[Path | str] = None,
    verbose: int = 1,
) -> pd.DataFrame:
    train_root = infer_train_root(data_folder)
    physio_root = infer_physiological_root(data_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if metadata_csv is None:
        metadata_csv = out_dir / "cache_metadata.csv"
    metadata_csv = Path(metadata_csv)

    if verbose:
        print(f"[INFO] train_root = {train_root}")
        print(f"[INFO] physio_root = {physio_root}")
        print(f"[INFO] out_dir = {out_dir}")
        print(f"[INFO] target_psg_sfreq = {target_psg_sfreq}")
        print(f"[INFO] target_ann_sfreq = {target_ann_sfreq}")
        print(f"[INFO] overwrite = {overwrite}")

    psg_files = list_physiological_edfs(physio_root)
    if limit_files is not None:
        psg_files = psg_files[:int(limit_files)]

    if verbose:
        print(f"[INFO] physiological EDF count = {len(psg_files)}")

    basename_index = build_index_by_basename(train_root)
    if verbose:
        print(f"[INFO] annotation/EDF basename index size = {len(basename_index)}")

    rows: List[Dict[str, object]] = []
    if metadata_csv.exists():
        try:
            old = pd.read_csv(metadata_csv)
            rows.extend(old.to_dict(orient="records"))
            if verbose:
                print(f"[INFO] existing metadata rows loaded = {len(old)}")
        except Exception:
            if verbose:
                print("[WARN] could not load existing metadata CSV; continuing fresh in-memory")

    done_names = set()
    for r in rows:
        if str(r.get("status", "")) in {"ok", "skip_existing"}:
            done_names.add(str(r.get("file_name", "")))

    processed_count = 0
    new_rows: List[Dict[str, object]] = []

    for i, psg_edf in enumerate(psg_files, start=1):
        file_name = psg_edf.name
        if (file_name in done_names) and not overwrite:
            if verbose:
                print(f"[SKIP] {i}/{len(psg_files)} -> {file_name} already in metadata")
            continue

        if verbose:
            print(f"[INFO] {i}/{len(psg_files)} -> {file_name}")

        row = process_one_file(
            psg_edf=psg_edf,
            basename_index=basename_index,
            out_dir=out_dir,
            overwrite=overwrite,
            target_psg_sfreq=target_psg_sfreq,
            target_ann_sfreq=target_ann_sfreq,
            read_verbose=read_verbose,
        )
        new_rows.append(row)
        processed_count += 1

        if verbose:
            status = row.get("status", "unknown")
            elapsed = row.get("elapsed_sec_total", np.nan)
            print(f"  [{status.upper()}] total={elapsed:.2f}s")

        if (processed_count % flush_every == 0) or (i == len(psg_files)):
            df = pd.DataFrame(rows + new_rows)
            df.to_csv(metadata_csv, index=False)
            if verbose:
                print(f"  [FLUSH] metadata -> {metadata_csv}")

    final_df = pd.DataFrame(rows + new_rows)
    final_df.to_csv(metadata_csv, index=False)

    if verbose:
        print("\n[DONE] Cache build complete.")
        if len(final_df) > 0 and "status" in final_df.columns:
            print(final_df["status"].value_counts(dropna=False).to_string())

    return final_df


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported and called via build_cache(...), "
        "not run as a hardcoded standalone script."
    )
