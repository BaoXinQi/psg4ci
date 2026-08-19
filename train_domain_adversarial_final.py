#!/usr/bin/env python3
"""Train the all-record weak domain-adversarial sequence model for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sequence_ci_model import FullNightEmbeddingDataset, LocalFullNightTransformer, collate_full_nights
from train_domain_adversarial_loso import (
    build_model,
    load_frame,
    predict,
    set_seed,
    site_class_weights,
    train_epoch,
    zeros,
)
from train_sequence_ci_loso import evaluate_scores, make_loader


PROTOCOL = "dynamic_full_training_m2_weak_domain_adversarial_v19"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def inference_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("site_head.")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_path = args.output_dir / "_SUCCESS"
    if success_path.is_file():
        print(success_path.read_text().strip(), flush=True)
        return

    frame = load_frame(args.manifest, args.embedding_manifest)
    record_count = int(len(frame))
    positive_count = int(frame["_label"].sum())
    if record_count < 20 or positive_count <= 0 or positive_count >= record_count:
        raise RuntimeError("Invalid all-data cohort")
    site_names = tuple(sorted(frame["SiteID"].unique()))
    if len(site_names) < 2:
        raise RuntimeError("Domain-adversarial training requires at least two sites")
    site_to_index = {name: index for index, name in enumerate(site_names)}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.fixed_epochs < 1:
        raise ValueError("--fixed-epochs must be positive")

    set_seed(args.seed)
    model = build_model(len(site_names), device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(frame["_label"].sum())
    positive_weight = (len(frame) - positives) / positives
    class_weights = site_class_weights(frame, site_names, device)
    loader, sampler = make_loader(
        frame,
        zeros(frame),
        args.batch_size,
        args.num_workers,
        args.seed + 17,
        True,
    )
    if sampler is None:
        raise RuntimeError("Missing training sampler")

    history: list[dict[str, Any]] = []
    start_epoch = 1
    existing = sorted(args.output_dir.glob("checkpoint_epoch_*.pt"))
    if existing:
        checkpoint = torch.load(existing[-1], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, args.fixed_epochs + 1):
        metrics = train_epoch(
            model,
            loader,
            sampler,
            optimizer,
            device,
            epoch,
            positive_weight,
            site_to_index,
            class_weights,
            args.gradient_clip,
        )
        history.append({"epoch": int(epoch), **metrics})
        checkpoint_path = args.output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        atomic_torch_save(
            {
                "protocol": PROTOCOL,
                "epoch": int(epoch),
                "epochs": args.fixed_epochs,
                "seed": args.seed,
                "model_state": cpu_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    evaluation_loader = torch.utils.data.DataLoader(
        FullNightEmbeddingDataset(frame, zeros(frame)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_full_nights,
    )
    scores = predict(model, evaluation_loader, device)
    if len(scores) != record_count or not np.isfinite(scores).all():
        raise FloatingPointError("Final training predictions are incomplete or non-finite")

    stripped_state = inference_state_dict(model)
    inference_model = LocalFullNightTransformer(
        demographic_dimension=1,
        use_demographics=False,
        d_model=256,
        dropout=0.15,
    )
    inference_model.load_state_dict(stripped_state, strict=True)
    final_path = args.output_dir / "final_model.pt"
    atomic_torch_save(
        {
            "protocol": PROTOCOL,
            "model_name": "raw_sequence",
            "records": record_count,
            "positives": positive_count,
            "epochs": args.fixed_epochs,
            "seed": args.seed,
            "model_state": stripped_state,
            "history": history,
            "training_configuration": {
                "main_loss": "weighted BCE + 0.15 same-site age pairwise",
                "site_adversary": "training-only class-balanced head",
                "gradient_reversal_strength": 0.05,
                "site_sampling": "80% natural + 20% site-balanced",
                "encoder": "frozen E1 FP32 EMA checkpoint",
                "embedding_dimension": 192,
                "demographics": False,
            },
            "integrity_only_in_sample_metrics": evaluate_scores(frame, scores),
        },
        final_path,
    )
    np.savez_compressed(
        args.output_dir / "training_predictions.npz",
        record_id=frame["record_id"].astype(str).to_numpy(),
        scores=scores.astype(np.float32),
    )
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "records": record_count,
        "positives": positive_count,
        "sites": frame["SiteID"].value_counts().sort_index().to_dict(),
        "epochs": args.fixed_epochs,
        "seed": args.seed,
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256(final_path),
        "history": history,
        "integrity_only_in_sample_metrics": evaluate_scores(frame, scores),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    success_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
