#!/usr/bin/env python3
"""Strict LOSO test for weak site-adversarial whole-night training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from domain_adversarial_sequence_model import (
    DOMAIN_REVERSAL_STRENGTH,
    DomainAdversarialFullNightTransformer,
)
from train_sequence_ci_loso import (
    deterministic_inner_split,
    evaluate_scores,
    make_loader,
    move_batch,
    predict,
    set_seed,
    training_loss,
)


SITES = ("I0002", "I0006", "S0001")
PAIRWISE_WEIGHT = 0.15
PAIR_SCOPE = "same_site"


def load_frame(manifest: Path, embedding_manifest: Path) -> pd.DataFrame:
    frame = pd.read_parquet(manifest).copy()
    embeddings = pd.read_parquet(embedding_manifest).copy()
    frame["record_id"] = frame["record_id"].astype(str)
    embeddings["record_id"] = embeddings["record_id"].astype(str)
    frame = frame.merge(
        embeddings[
            [
                "record_id",
                "shard_path",
                "group_key",
                "embedding_count",
                "complete_epoch_count",
            ]
        ],
        on="record_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_embedding"),
    )
    if len(frame) != frame["record_id"].nunique() or len(frame) < 20:
        raise RuntimeError(f"Invalid dynamic training cohort: {len(frame)} records")
    frame["SiteID"] = frame["SiteID"].astype(str)
    if frame["SiteID"].nunique() < 2:
        raise RuntimeError("Domain-adversarial training requires at least two sites")
    label_column = "label" if "label" in frame else "Cognitive_Impairment"
    frame["_label"] = pd.to_numeric(frame[label_column], errors="raise").astype(int)
    frame["_age"] = pd.to_numeric(frame["Age"], errors="raise").astype(float)
    if set(frame["_label"].unique()) != {0, 1}:
        raise RuntimeError("Dynamic training cohort must contain both labels")
    return frame.reset_index(drop=True)


def zeros(frame: pd.DataFrame) -> np.ndarray:
    return np.zeros((len(frame), 1), dtype=np.float32)


def build_model(
    site_count: int, device: torch.device
) -> DomainAdversarialFullNightTransformer:
    return DomainAdversarialFullNightTransformer(
        site_count=site_count,
        demographic_dimension=1,
        d_model=256,
        dropout=0.15,
        reversal_strength=DOMAIN_REVERSAL_STRENGTH,
    ).to(device)


def site_class_weights(
    frame: pd.DataFrame, site_names: tuple[str, ...], device: torch.device
) -> torch.Tensor:
    counts = frame["SiteID"].value_counts()
    values = np.asarray([float(counts[name]) for name in site_names])
    weights = len(frame) / (len(site_names) * values)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_epoch(
    model: DomainAdversarialFullNightTransformer,
    loader: DataLoader,
    sampler: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    positive_weight: float,
    site_to_index: dict[str, int],
    site_weights: torch.Tensor,
    gradient_clip: float,
) -> dict[str, Any]:
    model.train()
    sampler.set_epoch(epoch)
    totals = {
        "loss": 0.0,
        "main_loss": 0.0,
        "bce": 0.0,
        "pairwise": 0.0,
        "site_loss": 0.0,
        "site_correct": 0.0,
        "records": 0.0,
        "valid_pairs": 0.0,
        "batches": 0.0,
        "batches_with_pairs": 0.0,
    }
    pair_counts: dict[str, int] = {}
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("site_head.")
    ]
    site_parameters = list(model.site_head.parameters())
    started = time.time()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        main_logits, site_logits = model(
            batch["blocks"],
            batch["window_mask"],
            batch["demographics"],
            return_site=True,
        )
        main_loss, bce, pairwise, valid_pair_count, batch_pair_counts = training_loss(
            main_logits,
            batch["label"],
            batch["age"],
            batch["site_id"],
            positive_weight,
            PAIRWISE_WEIGHT,
            PAIR_SCOPE,
        )
        site_target = torch.as_tensor(
            [site_to_index[name] for name in batch["site_id"]],
            dtype=torch.long,
            device=device,
        )
        site_loss = F.cross_entropy(site_logits, site_target, weight=site_weights)
        loss = main_loss + site_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite domain loss: {batch['record_id']}")
        loss.backward()
        base_gradient_norm = nn.utils.clip_grad_norm_(
            base_parameters, gradient_clip
        )
        site_gradient_norm = nn.utils.clip_grad_norm_(
            site_parameters, gradient_clip
        )
        if not torch.isfinite(base_gradient_norm) or not torch.isfinite(
            site_gradient_norm
        ):
            raise FloatingPointError("Non-finite domain-adversarial gradient")
        optimizer.step()
        count = int(len(batch["label"]))
        totals["loss"] += float(loss.detach()) * count
        totals["main_loss"] += float(main_loss.detach()) * count
        totals["bce"] += float(bce.detach()) * count
        totals["pairwise"] += float(pairwise.detach()) * count
        totals["site_loss"] += float(site_loss.detach()) * count
        totals["site_correct"] += int((site_logits.argmax(dim=1) == site_target).sum())
        totals["records"] += count
        totals["valid_pairs"] += valid_pair_count
        totals["batches"] += 1
        totals["batches_with_pairs"] += int(valid_pair_count > 0)
        for name, value in batch_pair_counts.items():
            pair_counts[name] = pair_counts.get(name, 0) + int(value)
    denominator = max(totals["records"], 1.0)
    return {
        "epoch": int(epoch),
        "loss": totals["loss"] / denominator,
        "main_loss": totals["main_loss"] / denominator,
        "bce": totals["bce"] / denominator,
        "pairwise": totals["pairwise"] / denominator,
        "site_loss": totals["site_loss"] / denominator,
        "site_accuracy": totals["site_correct"] / denominator,
        "records": int(totals["records"]),
        "valid_pairs": int(totals["valid_pairs"]),
        "batches": int(totals["batches"]),
        "batch_pair_coverage": float(
            totals["batches_with_pairs"] / max(totals["batches"], 1.0)
        ),
        "pair_counts": pair_counts,
        "last_base_gradient_norm": float(base_gradient_norm.detach()),
        "last_site_gradient_norm": float(site_gradient_norm.detach()),
        "elapsed_seconds": time.time() - started,
    }


def select_epoch(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> int:
    site_names = tuple(sorted(training["SiteID"].unique()))
    site_to_index = {name: index for index, name in enumerate(site_names)}
    training_loader, sampler = make_loader(
        training,
        zeros(training),
        args.batch_size,
        args.num_workers,
        args.seed,
        True,
    )
    validation_loader, _ = make_loader(
        validation,
        zeros(validation),
        args.batch_size,
        args.num_workers,
        args.seed,
        False,
    )
    if sampler is None:
        raise RuntimeError("Missing domain training sampler")
    set_seed(args.seed)
    model = build_model(len(site_names), device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(training["_label"].sum())
    positive_weight = (len(training) - positives) / max(positives, 1.0)
    class_weights = site_class_weights(training, site_names, device)
    best_score = -math.inf
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        training_metrics = train_epoch(
            model,
            training_loader,
            sampler,
            optimizer,
            device,
            epoch,
            positive_weight,
            site_to_index,
            class_weights,
            args.gradient_clip,
        )
        scores = predict(model, validation_loader, device)
        validation_metrics = evaluate_scores(validation, scores)
        selection = float(validation_metrics["age_conditioned_auroc"])
        row = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "selection": selection,
        }
        history.append(row)
        print(json.dumps({"phase": "inner", **row}, sort_keys=True), flush=True)
        if selection > best_score + 1e-4:
            best_score = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite domain-adversarial checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inner_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )
    torch.save(
        {
            "model_name": "raw_sequence_domain_adversarial",
            "model_state": best_state,
            "selected_epoch": int(best_epoch),
            "site_names": site_names,
            "reversal_strength": DOMAIN_REVERSAL_STRENGTH,
        },
        output_dir / "inner_best.pt",
    )
    return best_epoch


def refit_and_predict(
    training: pd.DataFrame,
    held_out: pd.DataFrame,
    selected_epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> np.ndarray:
    site_names = tuple(sorted(training["SiteID"].unique()))
    site_to_index = {name: index for index, name in enumerate(site_names)}
    training_loader, sampler = make_loader(
        training,
        zeros(training),
        args.batch_size,
        args.num_workers,
        args.seed + 17,
        True,
    )
    held_loader, _ = make_loader(
        held_out,
        zeros(held_out),
        args.batch_size,
        args.num_workers,
        args.seed,
        False,
    )
    if sampler is None:
        raise RuntimeError("Missing domain refit sampler")
    set_seed(args.seed)
    model = build_model(len(site_names), device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(training["_label"].sum())
    positive_weight = (len(training) - positives) / max(positives, 1.0)
    class_weights = site_class_weights(training, site_names, device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, selected_epoch + 1):
        metrics = train_epoch(
            model,
            training_loader,
            sampler,
            optimizer,
            device,
            epoch,
            positive_weight,
            site_to_index,
            class_weights,
            args.gradient_clip,
        )
        history.append(metrics)
        print(json.dumps({"phase": "refit", **metrics}, sort_keys=True), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "raw_sequence_domain_adversarial",
            "model_state": model.state_dict(),
            "selected_epoch": int(selected_epoch),
            "site_names": site_names,
            "reversal_strength": DOMAIN_REVERSAL_STRENGTH,
        },
        output_dir / "final_model.pt",
    )
    (output_dir / "refit_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )
    return predict(model, held_loader, device).astype(np.float64)


def aggregate(metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    values = [float(metrics[site]["age_conditioned_auroc"]) for site in SITES]
    return {
        "macro_site_age_conditioned_auroc": float(np.mean(values)),
        "worst_site_age_conditioned_auroc": float(np.min(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.15)
    parser.add_argument("--matched-baseline-epochs", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for marker in ("_SUCCESS", "_FAILED"):
        (args.output_dir / marker).unlink(missing_ok=True)
    frame = load_frame(args.manifest, args.embedding_manifest)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    baseline_metrics: dict[str, dict[str, float | int]] = {}
    candidate_metrics: dict[str, dict[str, float | int]] = {}
    prediction_rows: list[pd.DataFrame] = []
    fold_audits: dict[str, Any] = {}
    for held_out_site in SITES:
        fold_dir = args.output_dir / f"held_{held_out_site}"
        outer_training = frame.loc[~frame["SiteID"].eq(held_out_site)].copy()
        held_out = frame.loc[frame["SiteID"].eq(held_out_site)].copy()
        inner_training, inner_validation = deterministic_inner_split(
            outer_training, args.inner_validation_fraction, args.seed
        )
        baseline_fold_dir = args.baseline_root / "raw_sequence" / f"held_{held_out_site}"
        baseline_fold_metrics = json.loads(
            (baseline_fold_dir / "fold_metrics.json").read_text()
        )
        if args.matched_baseline_epochs:
            selected_epoch = int(baseline_fold_metrics["selected_epoch"])
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "epoch_source.json").write_text(
                json.dumps(
                    {
                        "selected_epoch": selected_epoch,
                        "source": str(baseline_fold_dir / "fold_metrics.json"),
                        "held_out_metrics_used": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            selected_epoch = select_epoch(
                inner_training.reset_index(drop=True),
                inner_validation.reset_index(drop=True),
                args,
                device,
                fold_dir,
            )
        scores = refit_and_predict(
            outer_training.reset_index(drop=True),
            held_out.reset_index(drop=True),
            selected_epoch,
            args,
            device,
            fold_dir,
        )
        candidate_fold_metrics = evaluate_scores(held_out, scores)
        baseline_metrics[held_out_site] = baseline_fold_metrics
        candidate_metrics[held_out_site] = candidate_fold_metrics
        fold_audits[held_out_site] = {
            "selected_epoch": int(selected_epoch),
            "training_sites": sorted(outer_training["SiteID"].unique().tolist()),
            "outer_training_records": int(len(outer_training)),
            "inner_training_records": int(len(inner_training)),
            "inner_validation_records": int(len(inner_validation)),
            "held_out_records": int(len(held_out)),
            "epoch_source": (
                "matched_v3_inner_selection"
                if args.matched_baseline_epochs
                else "candidate_inner_selection"
            ),
        }
        prediction = held_out[
            ["record_id", "patient_id", "SiteID", "_label", "_age"]
        ].copy()
        prediction["domain_adversarial_score"] = scores
        prediction_rows.append(prediction)

    baseline_aggregate = aggregate(baseline_metrics)
    candidate_aggregate = aggregate(candidate_metrics)
    site_changes = {
        site: float(candidate_metrics[site]["age_conditioned_auroc"])
        - float(baseline_metrics[site]["age_conditioned_auroc"])
        for site in SITES
    }
    macro_change = (
        candidate_aggregate["macro_site_age_conditioned_auroc"]
        - baseline_aggregate["macro_site_age_conditioned_auroc"]
    )
    worst_change = (
        candidate_aggregate["worst_site_age_conditioned_auroc"]
        - baseline_aggregate["worst_site_age_conditioned_auroc"]
    )
    criteria = {
        "macro_gain_at_least_0.010": macro_change >= 0.010,
        "at_least_two_sites_improve": sum(value > 0 for value in site_changes.values()) >= 2,
        "worst_site_not_lower": worst_change >= 0.0,
        "no_site_drop_beyond_0.010": min(site_changes.values()) >= -0.010,
    }
    summary = {
        "protocol": {
            "candidate": "v3 plus training-only class-balanced site adversary",
            "reversal_strength": DOMAIN_REVERSAL_STRENGTH,
            "site_head_gradient_scale": 1.0,
            "backbone_site_gradient_scale": -DOMAIN_REVERSAL_STRENGTH,
            "main_loss": "weighted BCE + 0.15 same-site age pairwise",
            "selection_metric": "age_conditioned_auroc",
            "inference_requires_site": False,
            "held_out_used_for_training_or_selection": False,
            "epoch_protocol": (
                "matched v3 independently selected epochs"
                if args.matched_baseline_epochs
                else "candidate inner-validation selection"
            ),
        },
        "baseline": {"sites": baseline_metrics, **baseline_aggregate},
        "candidate": {"sites": candidate_metrics, **candidate_aggregate},
        "site_changes": site_changes,
        "macro_change": macro_change,
        "worst_change": worst_change,
        "gate": {"criteria": criteria, "passed": all(criteria.values())},
        "fold_audits": fold_audits,
    }
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        args.output_dir / "loso_predictions.parquet", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "_SUCCESS").write_text("complete\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
