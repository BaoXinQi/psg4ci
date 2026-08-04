#!/usr/bin/env python3
"""PhysioNet Challenge 2026 entry: D + CAISR + protected full-H residual."""

from __future__ import annotations

import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from helper_code import *  # noqa: F401,F403

import online_features
import train_large_structured_baselines_v1 as model_core


SCRIPT_DIR = Path(__file__).resolve().parent
PRETRAINED_DIR = SCRIPT_DIR / "pretrained_model"
MODEL_SUBDIR = "frozen_full_large"
ADAPTATION_EPOCHS = 1
ADAPTATION_LEARNING_RATE = 1e-6
ADAPTATION_BATCH_SIZE = 128
DEFAULT_THRESHOLD = 0.5


def _clean_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_binary(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0", "positive"}:
        return 1
    if text in {"false", "f", "no", "n", "0", "0.0", "negative"}:
        return 0
    return None


def _demographics_path(data_folder: Path) -> Path:
    candidates = [data_folder / "demographics.csv", data_folder / "training_set" / "demographics.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find demographics.csv under {data_folder}")


def _data_root(data_folder: Path) -> Path:
    if (data_folder / "physiological_data").is_dir():
        return data_folder
    if (data_folder / "training_set" / "physiological_data").is_dir():
        return data_folder / "training_set"
    raise FileNotFoundError(f"Could not find physiological_data under {data_folder}")


def _row_value(row: pd.Series | dict, names: list[str], required: bool = True) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip() != "":
            return row[name]
    if required:
        raise KeyError(f"Missing required field; tried {names}")
    return None


def _record_parts(row: pd.Series | dict) -> tuple[str, str, str, str]:
    patient_id = _clean_identifier(
        _row_value(row, ["BidsFolder", "bids_folder", "patient_id", "PatientID"])
    )
    site_id = _clean_identifier(_row_value(row, ["SiteID", "site_id", "site"]))
    session_id = _clean_identifier(
        _row_value(row, ["SessionID", "session_id", "session"])
    )
    record_id = f"{patient_id}_ses-{session_id}"
    return patient_id, site_id, session_id, record_id


def _record_paths(data_root: Path, site_id: str, record_id: str) -> tuple[Path, Path | None]:
    psg_path = data_root / "physiological_data" / site_id / f"{record_id}.edf"
    if not psg_path.is_file():
        matches = list((data_root / "physiological_data").glob(f"*/{record_id}.edf"))
        if len(matches) == 1:
            psg_path = matches[0]
        else:
            raise FileNotFoundError(f"PSG not found for {record_id}")
    caisr_path = (
        data_root
        / "algorithmic_annotations"
        / site_id
        / f"{record_id}_caisr_annotations.edf"
    )
    if not caisr_path.is_file():
        matches = list(
            (data_root / "algorithmic_annotations").glob(
                f"*/{record_id}_caisr_annotations.edf"
            )
        )
        caisr_path = matches[0] if len(matches) == 1 else None
    return psg_path, caisr_path


def _demographic_features(row: pd.Series | dict) -> dict[str, Any]:
    def value(names: list[str]) -> Any:
        output = _row_value(row, names, required=False)
        return np.nan if output is None else output

    return {
        "Age": value(["Age", "age"]),
        "BMI": value(["BMI", "bmi"]),
        "Sex": value(["Sex", "sex"]),
        "Race": value(["Race", "race"]),
        "Ethnicity": value(["Ethnicity", "ethnicity"]),
    }


def _extract_training_row(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    try:
        _, site_id, _, record_id = _record_parts(row)
        psg_path, caisr_path = _record_paths(Path(job["data_root"]), site_id, record_id)
        features = online_features.extract_caisr_features(psg_path, caisr_path, record_id)
        output = {**_demographic_features(row), **features}
        output.update(
            {
                "record_id": record_id,
                "label": int(job["label"]),
                "status": "ok",
                "error": "",
            }
        )
        return output
    except Exception as exception:
        return {
            **_demographic_features(row),
            "record_id": job.get("record_id", ""),
            "label": int(job["label"]),
            "status": "failed",
            "error": repr(exception),
        }


def _copy_pretrained(model_folder: Path) -> Path:
    if not (PRETRAINED_DIR / "_SUCCESS.json").is_file():
        raise FileNotFoundError(f"Packaged pretrained model is incomplete: {PRETRAINED_DIR}")
    destination = model_folder / MODEL_SUBDIR
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PRETRAINED_DIR, destination)
    return destination


def _load_schema(model_root: Path) -> dict[str, Any]:
    return json.loads((model_root / "feature_schema.json").read_text(encoding="utf-8"))


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame


def _load_anchor(seed_dir: Path, device: torch.device) -> tuple[Any, nn.Module]:
    preprocessor = joblib.load(seed_dir / "anchor_preprocessor.joblib")
    checkpoint = torch.load(seed_dir / "anchor_model.pt", map_location="cpu", weights_only=False)
    model = model_core.MLP(int(checkpoint["input_dimension"]), dropout=0.30).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return preprocessor, model


def _adapt_anchor_models(
    model_root: Path,
    feature_frame: pd.DataFrame,
    labels: np.ndarray,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    device = torch.device("cpu")
    columns = schema["anchor_numeric_columns"] + schema["anchor_categorical_columns"]
    feature_frame = _ensure_columns(feature_frame, columns)
    metadata = json.loads((model_root / "metadata.json").read_text(encoding="utf-8"))
    rows = []
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    pos_weight = float(negatives / positives) if positives > 0 and negatives > 0 else 1.0

    for seed in metadata["seeds"]:
        seed_dir = model_root / f"seed_{seed}"
        preprocessor, model = _load_anchor(seed_dir, device)
        matrix = np.asarray(preprocessor.transform(feature_frame[columns]), dtype=np.float32)
        features = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        targets = torch.as_tensor(labels, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            before = model(features).cpu().numpy()

        for parameter in model.parameters():
            parameter.requires_grad = False
        final_layer = model.network[-1]
        for parameter in final_layer.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            final_layer.parameters(), lr=ADAPTATION_LEARNING_RATE, weight_decay=0.0
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 900000)
        model.eval()
        losses = []
        for epoch in range(ADAPTATION_EPOCHS):
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(labels), ADAPTATION_BATCH_SIZE):
                indices = order[start : start + ADAPTATION_BATCH_SIZE]
                optimizer.zero_grad(set_to_none=True)
                logits = model(features[indices])
                loss = criterion(logits, targets[indices])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            after = model(features).cpu().numpy()
        delta = after - before
        checkpoint = torch.load(seed_dir / "anchor_model.pt", map_location="cpu", weights_only=False)
        checkpoint["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        checkpoint["adaptation"] = {
            "records": int(len(labels)),
            "epochs": ADAPTATION_EPOCHS,
            "learning_rate": ADAPTATION_LEARNING_RATE,
            "mean_absolute_logit_delta": float(np.mean(np.abs(delta))),
            "max_absolute_logit_delta": float(np.max(np.abs(delta))),
        }
        torch.save(checkpoint, seed_dir / "anchor_model.pt")
        rows.append(
            {
                "seed": int(seed),
                "records": int(len(labels)),
                "epochs": ADAPTATION_EPOCHS,
                "mean_loss": float(np.mean(losses)),
                "mean_absolute_logit_delta": float(np.mean(np.abs(delta))),
                "max_absolute_logit_delta": float(np.max(np.abs(delta))),
            }
        )
    return rows


def train_model(data_folder, model_folder, verbose):
    data_folder = Path(data_folder)
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)
    model_root = _copy_pretrained(model_folder)
    schema = _load_schema(model_root)

    demographics = pd.read_csv(_demographics_path(data_folder))
    data_root = _data_root(data_folder)
    jobs = []
    for row in demographics.to_dict(orient="records"):
        label = _parse_binary(
            _row_value(
                row,
                ["Cognitive_Impairment", "cognitive_impairment", "label"],
                required=False,
            )
        )
        if label is None:
            continue
        try:
            _, _, _, record_id = _record_parts(row)
        except Exception:
            record_id = ""
        jobs.append(
            {"row": row, "label": label, "record_id": record_id, "data_root": str(data_root)}
        )
    if not jobs:
        raise ValueError("No labeled training records were found")

    workers = max(1, min(8, os.cpu_count() or 1))
    extracted = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_training_row, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            extracted.append(future.result())
            if verbose and (completed == 1 or completed == len(futures) or completed % 250 == 0):
                print(f"CAISR adaptation features: {completed}/{len(futures)}", flush=True)

    feature_frame = pd.DataFrame(extracted)
    failures = feature_frame[feature_frame["status"] != "ok"].copy()
    feature_frame = feature_frame[feature_frame["status"] == "ok"].reset_index(drop=True)
    required_successes = max(1, int(math.ceil(0.90 * len(jobs))))
    if len(feature_frame) < required_successes:
        raise RuntimeError(
            f"Too few adaptation records succeeded: {len(feature_frame)}/{len(jobs)}"
        )
    labels = feature_frame["label"].to_numpy(dtype=int)
    adaptation_rows = _adapt_anchor_models(model_root, feature_frame, labels, schema)

    pd.DataFrame(adaptation_rows).to_csv(model_folder / "adaptation_history.csv", index=False)
    failures.to_csv(model_folder / "adaptation_failures.csv", index=False)
    (model_folder / "training_metadata.json").write_text(
        json.dumps(
            {
                "labeled_records": len(jobs),
                "adaptation_records": len(feature_frame),
                "failed_records": len(failures),
                "adaptation_epochs": ADAPTATION_EPOCHS,
                "adaptation_learning_rate": ADAPTATION_LEARNING_RATE,
                "psg_encoder_and_residual_frozen": True,
                "anchor_trainable_part": "final_linear_layer",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved adapted model to {model_root}", flush=True)


def _load_seed_model(seed_dir: Path, device: torch.device, group_order: list[str]) -> dict[str, Any]:
    anchor_preprocessor, anchor_model = _load_anchor(seed_dir, device)
    projectors = {
        group_name: joblib.load(seed_dir / f"{group_name}_projector.joblib")
        for group_name in group_order
    }
    checkpoint = torch.load(
        seed_dir / "balanced_pca_residual_model.pt", map_location="cpu", weights_only=False
    )
    residual = model_core.ProtectedResidualHead(
        int(checkpoint["input_dimension"]), dropout=0.30, initial_gate=0.10
    ).to(device)
    residual.load_state_dict(checkpoint["state_dict"], strict=True)
    anchor_model.eval()
    residual.eval()
    return {
        "anchor_preprocessor": anchor_preprocessor,
        "anchor_model": anchor_model,
        "projectors": projectors,
        "residual_model": residual,
    }


def load_model(model_folder, verbose):
    model_root = Path(model_folder) / MODEL_SUBDIR
    metadata = json.loads((model_root / "metadata.json").read_text(encoding="utf-8"))
    schema = _load_schema(model_root)
    device = torch.device("cpu")
    seeds = [
        _load_seed_model(model_root / f"seed_{seed}", device, schema["group_order"])
        for seed in metadata["seeds"]
    ]
    if verbose:
        print(f"Loaded {len(seeds)} frozen Large models", flush=True)
    return {
        "model_root": model_root,
        "metadata": metadata,
        "schema": schema,
        "device": device,
        "seeds": seeds,
        "demographics_path": None,
        "demographics": None,
        "data_root": None,
    }


def _lookup_demographics(state: dict[str, Any], data_folder: Path, record: dict) -> pd.Series:
    path = _demographics_path(data_folder)
    if state["demographics_path"] != str(path):
        state["demographics"] = pd.read_csv(path)
        state["demographics_path"] = str(path)
        state["data_root"] = _data_root(data_folder)
    patient_id, _, session_id, _ = _record_parts(record)
    frame = state["demographics"]
    mask = frame["BidsFolder"].astype(str).eq(patient_id)
    if "SessionID" in frame:
        mask &= frame["SessionID"].map(_clean_identifier).eq(session_id)
    if not mask.any():
        raise KeyError(f"Demographics row not found for {patient_id}/session {session_id}")
    return frame.loc[mask].iloc[0]


def _predict_from_features(state: dict[str, Any], frame: pd.DataFrame) -> float:
    schema = state["schema"]
    anchor_columns = schema["anchor_numeric_columns"] + schema["anchor_categorical_columns"]
    all_columns = list(anchor_columns)
    for group_name in schema["group_order"]:
        all_columns.extend(schema["group_columns"][group_name])
    frame = _ensure_columns(frame, list(dict.fromkeys(all_columns)))
    logits = []
    with torch.no_grad():
        for seed in state["seeds"]:
            anchor_matrix = np.asarray(
                seed["anchor_preprocessor"].transform(frame[anchor_columns]), dtype=np.float32
            )
            anchor_tensor = torch.as_tensor(anchor_matrix, dtype=torch.float32)
            anchor_logit = seed["anchor_model"](anchor_tensor)
            projected = [
                seed["projectors"][group_name].transform(frame)
                for group_name in schema["group_order"]
            ]
            balanced = torch.as_tensor(
                np.concatenate(projected, axis=1), dtype=torch.float32
            )
            correction, _ = seed["residual_model"](balanced)
            logits.append(float((anchor_logit + correction)[0]))
    return float(np.mean(logits))


def run_model(model, record, data_folder, verbose):
    state = model
    data_folder = Path(data_folder)
    row = _lookup_demographics(state, data_folder, record)
    _, site_id, _, record_id = _record_parts(record)
    psg_path, caisr_path = _record_paths(Path(state["data_root"]), site_id, record_id)
    try:
        psg_features, caisr_features = online_features.extract_all_features(
            psg_path, caisr_path, record_id, site_id
        )
    except Exception as exception:
        if verbose:
            print(f"PSG branch failed for {record_id}; using structured fallback: {exception}")
        psg_features = {}
        caisr_features = online_features.extract_caisr_features(psg_path, caisr_path, record_id)
    feature_row = {**_demographic_features(row), **caisr_features, **psg_features}
    logit = _predict_from_features(state, pd.DataFrame([feature_row]))
    probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
    return bool(probability >= DEFAULT_THRESHOLD), float(probability)
