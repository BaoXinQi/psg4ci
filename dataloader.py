#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from channel_mapper import normalize_channel_name

MODALITIES = ["bas", "resp", "ekg", "emg"]
TOKEN_SEC = 5.0
PSG_SFREQ_EXPECTED = 128.0
ANN_SFREQ_EXPECTED = 2.0
PSG_SAMPLES_PER_TOKEN = int(round(PSG_SFREQ_EXPECTED * TOKEN_SEC))
ANN_SAMPLES_PER_TOKEN = int(round(ANN_SFREQ_EXPECTED * TOKEN_SEC))

CAISR_CHANNELS = [
    "arousal_caisr", "caisr_prob_no-ar", "caisr_prob_arous", "limb_caisr", "resp_caisr",
    "stage_caisr", "caisr_prob_n3", "caisr_prob_n2", "caisr_prob_n1", "caisr_prob_r", "caisr_prob_w",
]

NUMERIC_COLS = ["Age", "BMI"]
CATEGORICAL_COLS = ["Sex", "Race", "Ethnicity", "SiteID"]
CATEGORY_LEVELS = {
    "Sex": ["Female", "Male", "Unknown"],
    "Race": ["Asian", "Black", "Others", "Unavailable", "White", "Unknown"],
    "Ethnicity": ["Hispanic", "Not Hispanic", "Unavailable", "Unknown"],
    "SiteID": ["I0002", "I0006", "S0001", "Unknown"],
}


def default_demographics_path(data_folder: Path | str) -> Path:
    data_folder = Path(data_folder)
    candidates = [
        data_folder / "training_set" / "demographics.csv",
        data_folder / "demographics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find demographics.csv under {data_folder}")


def default_cache_dir(model_folder: Path | str, cache_name: str = "cache_h5") -> Path:
    model_folder = Path(model_folder)
    return model_folder / cache_name


def decode_string_array(arr) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]



def load_demographics_table(demog_path: Path) -> pd.DataFrame:
    df = pd.read_csv(demog_path)
    if "file_name" not in df.columns:
        if "BidsFolder" in df.columns and "SessionID" in df.columns:
            df["file_name"] = df.apply(lambda r: f"{r['BidsFolder']}_ses-{int(r['SessionID'])}.edf", axis=1)
        else:
            raise ValueError("demographics table missing file_name")
    return df


def build_h5_inventory(cache_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(cache_dir / "**" / "*.h5"), recursive=True))
    return pd.DataFrame({
        "h5_path": paths,
        "file": [Path(p).name for p in paths],
        "file_name": [Path(p).stem + ".edf" for p in paths],
    })





def standardize_bool_label(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    if s in {"true", "1", "1.0", "t", "y", "yes"}:
        return True
    if s in {"false", "0", "0.0", "f", "n", "no"}:
        return False
    return bool(value)

def ann_to_token_level(arr: np.ndarray) -> np.ndarray:
    c, t = arr.shape
    n_tokens = t // ANN_SAMPLES_PER_TOKEN
    if n_tokens <= 0:
        return np.zeros((0, c), dtype=np.float32)
    arr = arr[:, : n_tokens * ANN_SAMPLES_PER_TOKEN]
    x = arr.reshape(c, n_tokens, ANN_SAMPLES_PER_TOKEN).mean(axis=2).transpose(1, 0)
    return x.astype(np.float32, copy=False)


def select_annotation_channels(token_arr: np.ndarray, ch_names: List[str], wanted_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx_map = {normalize_channel_name(c): i for i, c in enumerate(ch_names)}
    n = token_arr.shape[0]
    out = np.zeros((n, len(wanted_names)), dtype=np.float32)
    mask = np.zeros((len(wanted_names),), dtype=np.float32)
    for j, name in enumerate(wanted_names):
        key = normalize_channel_name(name)
        if key in idx_map:
            out[:, j] = token_arr[:, idx_map[key]].astype(np.float32, copy=False)
            mask[j] = 1.0
    return out, mask


class FixedDemographicsEncoder:
    def __init__(
        self,
        numeric_cols: List[str] = NUMERIC_COLS,
        categorical_cols: List[str] = CATEGORICAL_COLS,
        category_levels: Dict[str, List[str]] = CATEGORY_LEVELS,
    ):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.category_levels = {k: list(v) for k, v in category_levels.items()}
        self.numeric_means: Dict[str, float] = {}
        self.numeric_stds: Dict[str, float] = {}
        self.output_dim = (
            len(self.numeric_cols) +
            len(self.numeric_cols) +
            sum(len(self.category_levels[c]) for c in self.categorical_cols) +
            len(self.categorical_cols)
        )
        self.is_fitted = False

    def _is_missing_raw(self, value: Any) -> bool:
        if pd.isna(value):
            return True
        s = str(value).strip()
        return s == ""

    def _normalize_category(self, col: str, value: Any) -> str:
        if self._is_missing_raw(value):
            return "Unknown"
        s = str(value).strip()
        if s in self.category_levels[col]:
            return s
        return "Unknown"

    def fit(self, df: pd.DataFrame) -> "FixedDemographicsEncoder":
        for col in self.numeric_cols:
            x = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
            mean = float(x.mean()) if x.notna().any() else 0.0
            std = float(x.std()) if x.notna().any() else 1.0
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std
        self.is_fitted = True
        return self

    def transform_row(self, row: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Demographics encoder is not fitted.")

        feats: List[float] = []
        numeric_missing_flags: List[float] = []
        categorical_missing_flags: List[float] = []

        for col in self.numeric_cols:
            x = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            is_missing = float(pd.isna(x))
            numeric_missing_flags.append(is_missing)
            x = self.numeric_means[col] if pd.isna(x) else float(x)
            z = (x - self.numeric_means[col]) / self.numeric_stds[col]
            feats.append(np.float32(z))

        feats.extend(np.float32(v) for v in numeric_missing_flags)

        for col in self.categorical_cols:
            raw_val = row.get(col, np.nan)
            categorical_missing_flags.append(float(self._is_missing_raw(raw_val)))
            value = self._normalize_category(col, raw_val)
            levels = self.category_levels[col]
            for level in levels:
                feats.append(np.float32(1.0 if value == level else 0.0))

        feats.extend(np.float32(v) for v in categorical_missing_flags)

        out = np.asarray(feats, dtype=np.float32)
        if out.shape[0] != self.output_dim:
            raise ValueError(f"Unexpected demographics dim {out.shape[0]} != {self.output_dim}")
        return out


@dataclass
class SequenceSample:
    file_name: str
    site: str
    h5_path: str
    n_chunks: int
    token_mask: np.ndarray
    psg_chunks: Dict[str, np.ndarray]
    psg_channel_masks: Dict[str, np.ndarray]
    psg_channel_names: Dict[str, List[str]]
    caisr_chunks: np.ndarray
    caisr_channel_mask: np.ndarray
    has_caisr: np.ndarray
    demo_x: np.ndarray
    y: np.float32
    has_label: np.bool_


class SequenceDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path | str,
        demog_path: Path | str,
        require_caisr: bool = False,
        max_files: Optional[int] = None,
        demographics_encoder: Optional[FixedDemographicsEncoder] = None,
        fit_demographics_encoder: bool = False,
        chunk_tokens: int = 60,
    ):
        self.cache_dir = Path(cache_dir)
        self.demog_path = Path(demog_path)
        self.require_caisr = require_caisr
        self.chunk_tokens = int(chunk_tokens)

        demog = load_demographics_table(self.demog_path)
        inv = build_h5_inventory(self.cache_dir)
        df = inv.merge(demog, on="file_name", how="inner")

        if "Cognitive_Impairment" not in df.columns:
            raise ValueError("Merged demographics table missing Cognitive_Impairment")

        # Keep the original official label semantics in study_df for bookkeeping
        # and reporting, but standardize values robustly.
        df["Cognitive_Impairment"] = df["Cognitive_Impairment"].apply(standardize_bool_label)

        if max_files is not None:
            df = df.iloc[:int(max_files)].copy()

        rows = []
        for _, row in df.iterrows():
            with h5py.File(row["h5_path"], "r") as hf:
                psg_sfreq = float(hf["psg"].attrs["sfreq"])
                if abs(psg_sfreq - PSG_SFREQ_EXPECTED) > 1e-6:
                    continue

                psg_token_counts = {
                    m: int(hf[f"psg/{m}/data"].shape[1]) // PSG_SAMPLES_PER_TOKEN
                    for m in MODALITIES
                }
                positive_psg_counts = [n for n in psg_token_counts.values() if n > 0]
                if len(positive_psg_counts) == 0:
                    continue

                has_caisr_group = "annotations/caisr" in hf
                has_caisr = False
                caisr_n = np.inf
                if has_caisr_group:
                    has_caisr = bool(hf["annotations/caisr"].attrs.get("available", False))
                    if has_caisr and "annotations/caisr/data" in hf:
                        caisr_sfreq = float(hf["annotations/caisr"].attrs["sfreq"])
                        if abs(caisr_sfreq - ANN_SFREQ_EXPECTED) <= 1e-6:
                            caisr_n_val = int(hf["annotations/caisr/data"].shape[1]) // ANN_SAMPLES_PER_TOKEN
                            if caisr_n_val > 0:
                                caisr_n = caisr_n_val
                            else:
                                has_caisr = False
                        else:
                            has_caisr = False

                if self.require_caisr and not has_caisr:
                    continue

                usable_candidates = list(positive_psg_counts)
                if has_caisr and np.isfinite(caisr_n):
                    usable_candidates.append(int(caisr_n))

                usable_n = int(min(usable_candidates)) if len(usable_candidates) > 0 else 0
                if usable_n <= 0:
                    continue

            rec = row.to_dict()
            rec["usable_n_tokens"] = usable_n
            rec["has_caisr"] = bool(has_caisr)
            for m in MODALITIES:
                rec[f"{m}_tokens"] = psg_token_counts[m]
            rows.append(rec)

        self.study_df = pd.DataFrame(rows).reset_index(drop=True)

        if demographics_encoder is None:
            demographics_encoder = FixedDemographicsEncoder()
        self.demographics_encoder = demographics_encoder

        if fit_demographics_encoder:
            self.demographics_encoder.fit(self.study_df)

        if not self.demographics_encoder.is_fitted:
            raise ValueError("Demographics encoder is not fitted. Fit first or set fit_demographics_encoder=True.")

    def __len__(self) -> int:
        return len(self.study_df)

    def __getitem__(self, idx: int) -> SequenceSample:
        row = self.study_df.iloc[idx]
        usable_n = int(row["usable_n_tokens"])
        n_chunks = int(np.ceil(usable_n / self.chunk_tokens))

        file_name = row["file_name"]
        h5_path = row["h5_path"]

        psg_chunks: Dict[str, np.ndarray] = {}
        psg_channel_masks: Dict[str, np.ndarray] = {}
        psg_channel_names: Dict[str, List[str]] = {}

        token_mask = np.zeros((n_chunks, self.chunk_tokens), dtype=np.bool_)
        for j in range(n_chunks):
            st = j * self.chunk_tokens
            ed = min((j + 1) * self.chunk_tokens, usable_n)
            if ed > st:
                token_mask[j, :ed - st] = True

        caisr_chunks = np.zeros((n_chunks, self.chunk_tokens, len(CAISR_CHANNELS)), dtype=np.float32)
        caisr_channel_mask = np.zeros((len(CAISR_CHANNELS),), dtype=np.float32)
        has_caisr = np.asarray([1.0 if bool(row["has_caisr"]) else 0.0], dtype=np.float32)

        with h5py.File(h5_path, "r") as hf:
            site = str(hf.attrs.get("site", ""))

            for mod in MODALITIES:
                g = hf[f"psg/{mod}"]
                full_len = int(g["data"].shape[1]) if "data" in g else 0
                chs = decode_string_array(g["channels"][()]) if "channels" in g else []
                n_channels = len(chs)
                chunks = np.zeros((n_chunks, self.chunk_tokens, n_channels, PSG_SAMPLES_PER_TOKEN), dtype=np.float32)
                ch_mask = np.ones((n_channels,), dtype=np.bool_)

                if full_len > 0 and int(row.get(f"{mod}_tokens", 0)) > 0 and n_channels > 0:
                    arr = g["data"][:, : usable_n * PSG_SAMPLES_PER_TOKEN].astype(np.float32)
                    n_tokens = arr.shape[1] // PSG_SAMPLES_PER_TOKEN
                    if n_tokens > 0:
                        tok = arr[:, : n_tokens * PSG_SAMPLES_PER_TOKEN]
                        tok = tok.reshape(n_channels, n_tokens, PSG_SAMPLES_PER_TOKEN).transpose(1, 0, 2)
                        for j in range(n_chunks):
                            st = j * self.chunk_tokens
                            ed = min((j + 1) * self.chunk_tokens, n_tokens)
                            if ed > st:
                                chunks[j, :ed - st] = tok[st:ed]
                else:
                    ch_mask = np.zeros((n_channels,), dtype=np.bool_)

                psg_chunks[mod] = chunks
                psg_channel_masks[mod] = ch_mask
                psg_channel_names[mod] = chs

            if bool(row["has_caisr"]) and "annotations/caisr/data" in hf:
                g = hf["annotations/caisr"]
                full_len = int(g["data"].shape[1])
                if full_len > 0:
                    caisr_arr = g["data"][:, : usable_n * ANN_SAMPLES_PER_TOKEN].astype(np.float32)
                    caisr_chs = decode_string_array(g["channels"][()])
                    caisr_tok = ann_to_token_level(caisr_arr)
                    caisr_tok, caisr_channel_mask = select_annotation_channels(caisr_tok, caisr_chs, CAISR_CHANNELS)

                    for j in range(n_chunks):
                        st = j * self.chunk_tokens
                        ed = min((j + 1) * self.chunk_tokens, caisr_tok.shape[0])
                        if ed > st:
                            caisr_chunks[j, :ed - st] = caisr_tok[st:ed]

        demo_x = self.demographics_encoder.transform_row(row)
        # Retraining logic: invert the target so the model learns the opposite
        # direction internally; team_code.py will map the output back to the
        # official Challenge probability for Cognitive_Impairment.
        raw_label = standardize_bool_label(row["Cognitive_Impairment"])
        y = np.float32(1.0 - float(raw_label))
        has_label = np.bool_(True)

        return SequenceSample(
            file_name=file_name,
            site=site,
            h5_path=str(h5_path),
            n_chunks=n_chunks,
            token_mask=token_mask,
            psg_chunks=psg_chunks,
            psg_channel_masks=psg_channel_masks,
            psg_channel_names=psg_channel_names,
            caisr_chunks=caisr_chunks,
            caisr_channel_mask=caisr_channel_mask,
            has_caisr=has_caisr,
            demo_x=demo_x,
            y=y,
            has_label=has_label,
        )


def collate_fn(batch: List[SequenceSample]) -> Dict[str, Any]:
    bsz = len(batch)
    n_chunks_max = max(x.n_chunks for x in batch)
    chunk_tokens = batch[0].token_mask.shape[1]

    out: Dict[str, Any] = {
        "file_name": [x.file_name for x in batch],
        "site": [x.site for x in batch],
        "h5_path": [x.h5_path for x in batch],
        "n_chunks": torch.tensor([x.n_chunks for x in batch], dtype=torch.int64),
        "token_mask": torch.zeros((bsz, n_chunks_max, chunk_tokens), dtype=torch.bool),
        "chunk_mask": torch.zeros((bsz, n_chunks_max), dtype=torch.bool),
        "caisr_chunks": torch.zeros((bsz, n_chunks_max, chunk_tokens, len(CAISR_CHANNELS)), dtype=torch.float32),
        "caisr_channel_mask": torch.zeros((bsz, len(CAISR_CHANNELS)), dtype=torch.float32),
        "has_caisr": torch.zeros((bsz, 1), dtype=torch.float32),
        "demo_x": torch.zeros((bsz, batch[0].demo_x.shape[0]), dtype=torch.float32),
        "y": torch.zeros((bsz,), dtype=torch.float32),
        "has_label": torch.zeros((bsz,), dtype=torch.bool),
    }

    for mod in MODALITIES:
        c_max = max(x.psg_chunks[mod].shape[2] for x in batch)
        out[f"{mod}_chunks"] = torch.zeros((bsz, n_chunks_max, chunk_tokens, c_max, PSG_SAMPLES_PER_TOKEN), dtype=torch.float32)
        out[f"{mod}_channel_mask"] = torch.zeros((bsz, c_max), dtype=torch.bool)
        out[f"{mod}_channel_names"] = []

    for i, x in enumerate(batch):
        n = x.n_chunks
        out["token_mask"][i, :n] = torch.from_numpy(x.token_mask)
        out["chunk_mask"][i, :n] = True
        out["caisr_chunks"][i, :n] = torch.from_numpy(x.caisr_chunks)
        out["caisr_channel_mask"][i] = torch.from_numpy(x.caisr_channel_mask)
        out["has_caisr"][i] = torch.from_numpy(x.has_caisr)
        out["demo_x"][i] = torch.from_numpy(x.demo_x)
        out["y"][i] = torch.tensor(x.y, dtype=torch.float32)
        out["has_label"][i] = torch.tensor(x.has_label, dtype=torch.bool)

        for mod in MODALITIES:
            c_i = x.psg_chunks[mod].shape[2]
            if c_i > 0:
                out[f"{mod}_chunks"][i, :n, :, :c_i, :] = torch.from_numpy(x.psg_chunks[mod])
                out[f"{mod}_channel_mask"][i, :c_i] = torch.from_numpy(x.psg_channel_masks[mod].astype(np.bool_))
            out[f"{mod}_channel_names"].append(list(x.psg_channel_names[mod]))

    return out


if __name__ == "__main__":
    print("dataloader.py written.")
