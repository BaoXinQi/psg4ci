#!/usr/bin/env python3
"""Export resumable, record-grouped 30-second embeddings for all Large nights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch

from domain_robust_encoder_model import DomainRobustPsgEncoder
from train_domain_robust_encoder_full import load_or_build_index, load_signal_batch


VERSION = "domain_robust_encoder_full_embeddings_v1"


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_record(
    model: DomainRobustPsgEncoder,
    record: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    indices = np.asarray(record["eligible"], dtype=np.int32)
    if len(indices) == 0:
        return {
            "window_index": indices,
            "embedding": np.empty((0, model.embedding_dimension), dtype=np.float16),
            "stage_logits": np.empty((0, 5), dtype=np.float16),
            "event_logits": np.empty((0, 3), dtype=np.float16),
            "modality_valid": np.empty((0, 6), dtype=np.uint8),
        }
    signals = load_signal_batch(str(record["canonical_path"]), indices)
    embeddings: list[np.ndarray] = []
    stage_logits: list[np.ndarray] = []
    event_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            batch = {
                key: value[start:stop].to(device, non_blocking=True)
                for key, value in signals.items()
            }
            output = model(batch)
            tensors = {
                "embedding": output["embedding"],
                "stage_logits": output["stage_logits"],
                "event_logits": output["event_logits"],
            }
            for name, value in tensors.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(
                        f"Non-finite {name} in {record['record_id']} windows {start}:{stop}"
                    )
            embeddings.append(tensors["embedding"].cpu().numpy().astype(np.float16))
            stage_logits.append(tensors["stage_logits"].cpu().numpy().astype(np.float16))
            event_logits.append(tensors["event_logits"].cpu().numpy().astype(np.float16))
    return {
        "window_index": indices,
        "embedding": np.concatenate(embeddings),
        "stage_logits": np.concatenate(stage_logits),
        "event_logits": np.concatenate(event_logits),
        "modality_valid": np.asarray(record["modality_valid"], dtype=np.uint8)[indices],
    }


def write_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    if len(values) == 0:
        group.create_dataset(name, data=values)
        return
    first = min(max(len(values), 1), 1024)
    chunks = (first, *values.shape[1:])
    group.create_dataset(
        name,
        data=values,
        chunks=chunks,
        compression="lzf",
        shuffle=True,
    )


def export_shard(
    shard_index: int,
    rows: pd.DataFrame,
    records: Mapping[str, Mapping[str, Any]],
    model: DomainRobustPsgEncoder,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    checkpoint_hash: str,
) -> None:
    shard_path = output_dir / "shards" / f"shard_{shard_index:04d}.h5"
    sidecar_path = output_dir / "manifests" / f"shard_{shard_index:04d}.parquet"
    if shard_path.is_file() and sidecar_path.is_file():
        return
    temporary = shard_path.with_suffix(".h5.tmp")
    if temporary.exists():
        temporary.unlink()
    metadata: list[dict[str, Any]] = []
    with h5py.File(temporary, "w") as handle:
        handle.attrs["version"] = VERSION
        handle.attrs["checkpoint_sha256"] = checkpoint_hash
        handle.attrs["shard_index"] = shard_index
        for position, row in enumerate(rows.to_dict(orient="records")):
            record_id = str(row["record_id"])
            record = records[record_id]
            payload = encode_record(model, record, device, batch_size)
            group_key = f"record_{position:04d}"
            group = handle.create_group(group_key)
            group.attrs["record_id"] = record_id
            group.attrs["patient_id"] = str(row["patient_id"])
            group.attrs["site"] = str(row["SiteID"])
            group.attrs["complete_epoch_count"] = int(record["n_epochs"])
            for name, values in payload.items():
                write_dataset(group, name, values)
            metadata.append(
                {
                    "record_id": record_id,
                    "patient_id": str(row["patient_id"]),
                    "SiteID": str(row["SiteID"]),
                    "shard_index": shard_index,
                    "shard_path": str(shard_path),
                    "group_key": group_key,
                    "complete_epoch_count": int(record["n_epochs"]),
                    "embedding_count": int(len(payload["window_index"])),
                }
            )
        handle.flush()
    temporary.replace(shard_path)
    sidecar_tmp = sidecar_path.with_suffix(".parquet.tmp")
    pd.DataFrame(metadata).to_parquet(sidecar_tmp, index=False)
    sidecar_tmp.replace(sidecar_path)


def finalize(output_dir: Path, expected_records: int, checkpoint_hash: str) -> dict[str, Any]:
    sidecars = sorted((output_dir / "manifests").glob("shard_*.parquet"))
    if not sidecars:
        raise RuntimeError("No shard manifests were produced")
    manifest = pd.concat([pd.read_parquet(path) for path in sidecars], ignore_index=True)
    if len(manifest) != expected_records or manifest["record_id"].duplicated().any():
        raise RuntimeError(
            f"Incomplete export: records={len(manifest)} expected={expected_records}"
        )
    manifest = manifest.sort_values(["SiteID", "record_id"]).reset_index(drop=True)
    manifest.to_parquet(output_dir / "embedding_manifest.parquet", index=False)
    summary = {
        "status": "complete",
        "version": VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "records": int(len(manifest)),
        "embedding_windows": int(manifest["embedding_count"].sum()),
        "shards": int(manifest["shard_index"].nunique()),
        "site_records": manifest["SiteID"].value_counts().sort_index().astype(int).to_dict(),
        "site_windows": manifest.groupby("SiteID")["embedding_count"].sum().astype(int).to_dict(),
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "_SUCCESS").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--records-per-shard", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest).copy()
    manifest["record_id"] = manifest["record_id"].astype(str)
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["SiteID"] = manifest["SiteID"].astype(str)
    manifest = manifest.sort_values(["SiteID", "record_id"]).reset_index(drop=True)
    index = load_or_build_index(manifest, args.annotation_cache_dir, args.candidate_index)
    records = index["records"]
    if args.max_records > 0:
        manifest = manifest.iloc[: args.max_records].copy()

    checkpoint = load_checkpoint(args.checkpoint)
    state_key = "teacher_state" if args.state == "teacher" else "model_state"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DomainRobustPsgEncoder().to(device).eval()
    incompatible = model.load_state_dict(checkpoint[state_key], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite checkpoint parameter: {name}")

    output_dir = args.output_dir
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(args.checkpoint)
    shard_count = math.ceil(len(manifest) / args.records_per_shard)
    started = time.time()
    for shard_index in range(shard_count):
        low = shard_index * args.records_per_shard
        high = min(low + args.records_per_shard, len(manifest))
        export_shard(
            shard_index,
            manifest.iloc[low:high],
            records,
            model,
            device,
            args.batch_size,
            output_dir,
            checkpoint_hash,
        )
        print(
            json.dumps(
                {
                    "shard": shard_index + 1,
                    "shards": shard_count,
                    "records": high,
                    "elapsed_sec": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(json.dumps(finalize(output_dir, len(manifest), checkpoint_hash), sort_keys=True))


if __name__ == "__main__":
    main()
