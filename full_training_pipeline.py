"""End-to-end official-server reproduction of the V17/V14-Large method."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from full_training_constants import (
    E1_BATCH_SIZE,
    E1_EMA_DECAY,
    E1_EPOCHS,
    E1_EVAL_WINDOWS_PER_RECORD,
    E1_LEARNING_RATE,
    E1_NATURAL_WINDOWS,
    E1_RECORDS_PER_BATCH,
    E1_SEED,
    E1_WEIGHT_DECAY,
    E1_WINDOWS_PER_RECORD,
    MODEL_SUBDIR,
    RESIDUAL_BLEND_WEIGHT,
    SEQUENCE_BATCH_SIZE,
    SEQUENCE_EPOCHS,
    SEQUENCE_LEARNING_RATE,
    SEQUENCE_WEIGHT_DECAY,
)
from full_training_data import clear_workspace, prepare_training_data
from full_training_residuals import fit_and_write_residuals


SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL = "v18_official_full_training_reproduction_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_stage(name: str, command: list[str], workspace: Path, verbose: bool) -> float:
    log_path = workspace / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if verbose:
                print(f"[{name}] {line}", end="", flush=True)
        status = process.wait()
    if status != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"Stage {name} failed with status {status}:\n" + "\n".join(tail))
    return time.time() - started


def save_teacher_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    teacher_state = checkpoint.get("teacher_state")
    if not isinstance(teacher_state, dict) or not teacher_state:
        raise RuntimeError("E1 checkpoint does not contain an EMA teacher state")
    for name, value in teacher_state.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite E1 teacher parameter: {name}")
    payload = {
        "protocol": PROTOCOL,
        "model_state": {name: value.detach().cpu() for name, value in teacher_state.items()},
        "source_epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["args"]["seed"]),
        "variant": str(checkpoint["variant"]),
    }
    torch.save(payload, destination)
    return {
        "source_epoch": payload["source_epoch"],
        "seed": payload["seed"],
        "variant": payload["variant"],
    }


def production_settings() -> dict[str, int]:
    test_mode = os.environ.get("PSG4CI_V18_TEST_MODE", "").strip() == "1"
    if not test_mode:
        return {
            "e1_epochs": E1_EPOCHS,
            "max_records": 0,
            "sequence_epoch_cap": 0,
        }
    return {
        "e1_epochs": max(1, int(os.environ.get("PSG4CI_V18_TEST_E1_EPOCHS", "1"))),
        "max_records": max(24, int(os.environ.get("PSG4CI_V18_TEST_MAX_RECORDS", "48"))),
        "sequence_epoch_cap": max(
            1, int(os.environ.get("PSG4CI_V18_TEST_SEQUENCE_EPOCHS", "1"))
        ),
    }


def build_metadata(
    model_root: Path,
    frame: pd.DataFrame,
    preprocessing: dict[str, Any],
    e1: dict[str, Any],
    residuals: dict[str, Any],
    stage_times: dict[str, float],
) -> dict[str, Any]:
    sequence_files = [f"raw_sequence_seed_{seed}.pt" for seed in SEQUENCE_EPOCHS]
    files: dict[str, Any] = {}
    for filename in [
        "e1_encoder.pt",
        *sequence_files,
        "date_residual.json",
        "caisr_residual.json",
        "followup_residual.json",
    ]:
        path = model_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing final artifact: {path}")
        files[filename] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for seed in SEQUENCE_EPOCHS:
        filename = f"raw_sequence_seed_{seed}.pt"
        try:
            payload = torch.load(model_root / filename, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(model_root / filename, map_location="cpu")
        files[filename].update(
            {"seed": int(payload["seed"]), "epochs": int(payload["epochs"])}
        )
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "model": (
            "officially retrained E1 + three-member weak domain-adversarial Raw "
            "ensemble + 0.375 record-wise date, CAISR, and follow-up residual"
        ),
        "training_records": int(len(frame)),
        "training_positives": int(frame["label"].sum()),
        "training_sites": frame["SiteID"].value_counts().sort_index().astype(int).to_dict(),
        "encoder": e1,
        "sequence_files": sequence_files,
        "ensemble": "equal-weight mean of logits",
        "pair_scope": "same_site",
        "pairwise_weight": 0.15,
        "domain_adversary": {
            "training_only": True,
            "gradient_reversal_strength": 0.05,
            "inference_requires_site": False,
        },
        "creation_time_residual": {
            "file": "date_residual.json",
            "record_wise": True,
            "missing_behavior": "current-record EDF start time, then zero adjustment",
        },
        "caisr_residual": {
            "file": "caisr_residual.json",
            "features": 90,
            "record_wise": True,
            "missing_behavior": "zero adjustment",
        },
        "followup_residual": {
            "file": "followup_residual.json",
            "record_wise": True,
            "residual_blend_weight": RESIDUAL_BLEND_WEIGHT,
            "hidden_cohort_statistics": False,
        },
        "preprocessing": preprocessing,
        "residual_fit": residuals,
        "stage_elapsed_sec": stage_times,
        "files": files,
    }


def run_full_training(data_folder: Path, model_folder: Path, verbose: bool) -> Path:
    settings = production_settings()
    if settings["max_records"] == 0 and not torch.cuda.is_available():
        raise RuntimeError("Production V18 full training requires a CUDA GPU")

    workspace = Path(os.environ.get("PSG4CI_V18_WORKSPACE", "/tmp/psg4ci_v18_full_training"))
    clear_workspace(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    stage_times: dict[str, float] = {}
    total_started = time.time()

    preprocessing_started = time.time()
    manifest_path, frame, preprocessing = prepare_training_data(
        Path(data_folder),
        workspace,
        workers=min(16, max(1, os.cpu_count() or 1)),
        verbose=verbose,
        max_records=settings["max_records"],
    )
    stage_times["preprocessing"] = time.time() - preprocessing_started

    annotation_dir = workspace / "annotations"
    candidate_index = workspace / "index" / "candidate_index.pkl.gz"
    e1_dir = workspace / "e1"
    e1_command = [
        sys.executable,
        str(SCRIPT_DIR / "train_domain_robust_encoder_full.py"),
        "--manifest", str(manifest_path),
        "--annotation-cache-dir", str(annotation_dir),
        "--candidate-index", str(candidate_index),
        "--output-dir", str(e1_dir),
        "--variant", "e1",
        "--epochs", str(settings["e1_epochs"]),
        "--windows-per-record", str(E1_WINDOWS_PER_RECORD),
        "--natural-windows", str(E1_NATURAL_WINDOWS),
        "--eval-windows-per-record", str(E1_EVAL_WINDOWS_PER_RECORD),
        "--batch-size", str(E1_BATCH_SIZE),
        "--records-per-batch", str(E1_RECORDS_PER_BATCH),
        "--num-workers", "4",
        "--learning-rate", str(E1_LEARNING_RATE),
        "--weight-decay", str(E1_WEIGHT_DECAY),
        "--ema-decay", str(E1_EMA_DECAY),
        "--seed", str(E1_SEED),
        "--device", "cuda",
        "--overwrite",
    ]
    stage_times["e1_training"] = run_stage("e1_training", e1_command, workspace, verbose)
    e1_checkpoint = e1_dir / f"checkpoint_epoch_{settings['e1_epochs']:02d}.pt"
    if not e1_checkpoint.is_file():
        raise FileNotFoundError(f"Final E1 checkpoint was not created: {e1_checkpoint}")

    embeddings_dir = workspace / "embeddings"
    export_command = [
        sys.executable,
        str(SCRIPT_DIR / "export_full_embeddings.py"),
        "--manifest", str(manifest_path),
        "--annotation-cache-dir", str(annotation_dir),
        "--candidate-index", str(candidate_index),
        "--checkpoint", str(e1_checkpoint),
        "--state", "teacher",
        "--output-dir", str(embeddings_dir),
        "--records-per-shard", "100",
        "--batch-size", "64",
        "--device", "cuda",
    ]
    stage_times["embedding_export"] = run_stage(
        "embedding_export", export_command, workspace, verbose
    )
    embedding_manifest = embeddings_dir / "embedding_manifest.parquet"

    sequence_outputs: dict[int, Path] = {}
    for seed, production_epochs in SEQUENCE_EPOCHS.items():
        epochs = production_epochs
        if settings["sequence_epoch_cap"]:
            epochs = min(epochs, settings["sequence_epoch_cap"])
        output_dir = workspace / "sequence" / f"seed_{seed}"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "train_domain_adversarial_final.py"),
            "--manifest", str(manifest_path),
            "--embedding-manifest", str(embedding_manifest),
            "--output-dir", str(output_dir),
            "--batch-size", str(SEQUENCE_BATCH_SIZE),
            "--num-workers", "4",
            "--learning-rate", str(SEQUENCE_LEARNING_RATE),
            "--weight-decay", str(SEQUENCE_WEIGHT_DECAY),
            "--fixed-epochs", str(epochs),
            "--seed", str(seed),
            "--device", "cuda",
        ]
        stage_times[f"sequence_seed_{seed}"] = run_stage(
            f"sequence_seed_{seed}", command, workspace, verbose
        )
        sequence_outputs[seed] = output_dir / "final_model.pt"

    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)
    staging = model_folder / f".{MODEL_SUBDIR}.tmp"
    final_root = model_folder / MODEL_SUBDIR
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    e1_metadata = save_teacher_checkpoint(e1_checkpoint, staging / "e1_encoder.pt")
    for seed, source in sequence_outputs.items():
        if not source.is_file():
            raise FileNotFoundError(f"Sequence checkpoint was not created: {source}")
        shutil.copy2(source, staging / f"raw_sequence_seed_{seed}.pt")

    residual_started = time.time()
    residuals = fit_and_write_residuals(frame, staging)
    stage_times["residual_fitting"] = time.time() - residual_started
    e1_metadata.update(
        {
            "epochs": settings["e1_epochs"],
            "windows_per_record_per_epoch": E1_WINDOWS_PER_RECORD,
            "precision": "FP32",
            "deployed_state": "EMA teacher",
        }
    )
    metadata = build_metadata(
        staging, frame, preprocessing, e1_metadata, residuals, stage_times
    )
    metadata["total_elapsed_sec"] = time.time() - total_started
    metadata["test_mode"] = bool(settings["max_records"])
    (staging / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Loading every component is the final atomic-package gate.
    from raw_sequence_runtime import load_runtime

    runtime = load_runtime(staging, threads=2)
    if len(runtime["sequences"]) != len(SEQUENCE_EPOCHS):
        raise RuntimeError("Final runtime did not load all sequence members")
    del runtime
    if final_root.exists():
        shutil.rmtree(final_root)
    staging.replace(final_root)
    clear_workspace(workspace)
    return final_root
