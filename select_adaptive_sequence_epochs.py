#!/usr/bin/env python3
"""Select per-seed sequence budgets on the freshly trained E1 embeddings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from full_training_constants import (
    SEQUENCE_BATCH_SIZE,
    SEQUENCE_EPOCHS,
    SEQUENCE_LEARNING_RATE,
    SEQUENCE_SELECTION_INNER_FRACTION,
    SEQUENCE_SELECTION_MAX_EPOCHS,
    SEQUENCE_SELECTION_MAX_SITE_DROP,
    SEQUENCE_SELECTION_MAX_WORST_DROP,
    SEQUENCE_SELECTION_MIN_MACRO_GAIN,
    SEQUENCE_SELECTION_NEAR_BEST_AC,
    SEQUENCE_WEIGHT_DECAY,
)
from train_domain_adversarial_loso import (
    SITES,
    load_frame,
    refit_and_predict,
    select_epoch,
)
from train_sequence_ci_loso import deterministic_inner_split, evaluate_scores


def robust_epoch_from_history(
    history: list[dict[str, Any]], near_best_ac: float
) -> int:
    finite = [
        (int(row["epoch"]), float(row["selection"]))
        for row in history
        if np.isfinite(float(row["selection"]))
    ]
    if not finite:
        raise RuntimeError("No finite inner-validation AC values")
    best = max(value for _, value in finite)
    eligible = [epoch for epoch, value in finite if value >= best - near_best_ac]
    return int(min(eligible))


def median_epoch(values: list[int]) -> int:
    if not values:
        raise ValueError("No fold epochs were provided")
    return int(np.median(np.asarray(values, dtype=np.int64)))


def logit_mean_probabilities(member_probabilities: list[np.ndarray]) -> np.ndarray:
    if not member_probabilities:
        raise ValueError("No ensemble members were provided")
    matrix = np.column_stack(member_probabilities).astype(np.float64)
    matrix = np.clip(matrix, 1e-6, 1.0 - 1e-6)
    logits = np.log(matrix) - np.log1p(-matrix)
    mean_logit = logits.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -40.0, 40.0)))


def aggregate_site_metrics(site_metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    values = np.asarray(
        [float(site_metrics[site]["age_conditioned_auroc"]) for site in SITES],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite site AC values: {values.tolist()}")
    return {
        "macro_site_age_conditioned_auroc": float(values.mean()),
        "worst_site_age_conditioned_auroc": float(values.min()),
    }


def selection_gate(
    baseline_metrics: dict[str, dict[str, float | int]],
    candidate_metrics: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    baseline = aggregate_site_metrics(baseline_metrics)
    candidate = aggregate_site_metrics(candidate_metrics)
    site_deltas = {
        site: float(candidate_metrics[site]["age_conditioned_auroc"])
        - float(baseline_metrics[site]["age_conditioned_auroc"])
        for site in SITES
    }
    macro_delta = (
        candidate["macro_site_age_conditioned_auroc"]
        - baseline["macro_site_age_conditioned_auroc"]
    )
    worst_delta = (
        candidate["worst_site_age_conditioned_auroc"]
        - baseline["worst_site_age_conditioned_auroc"]
    )
    checks = {
        "macro_gain": macro_delta >= SEQUENCE_SELECTION_MIN_MACRO_GAIN,
        "two_sites_improve": sum(value > 0.0 for value in site_deltas.values()) >= 2,
        "worst_preserved": worst_delta >= -SEQUENCE_SELECTION_MAX_WORST_DROP,
        "no_site_collapse": min(site_deltas.values()) >= -SEQUENCE_SELECTION_MAX_SITE_DROP,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "site_deltas": site_deltas,
        "macro_delta": float(macro_delta),
        "worst_delta": float(worst_delta),
        "baseline": baseline,
        "candidate": candidate,
    }


def training_args(parsed: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=int(seed),
        batch_size=int(parsed.batch_size),
        num_workers=int(parsed.num_workers),
        learning_rate=float(parsed.learning_rate),
        weight_decay=float(parsed.weight_decay),
        gradient_clip=float(parsed.gradient_clip),
        max_epochs=int(parsed.max_epochs),
        minimum_epochs=int(parsed.max_epochs),
        patience=int(parsed.max_epochs),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=SEQUENCE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=SEQUENCE_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=SEQUENCE_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=SEQUENCE_SELECTION_MAX_EPOCHS)
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        default=SEQUENCE_SELECTION_INNER_FRACTION,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for marker in ("_SUCCESS", "_FAILED"):
        (args.output_dir / marker).unlink(missing_ok=True)
    frame = load_frame(args.manifest, args.embedding_manifest)
    if tuple(sorted(frame["SiteID"].unique())) != tuple(sorted(SITES)):
        raise RuntimeError(
            "Adaptive epoch selection requires the three locked training sites; "
            f"found {sorted(frame['SiteID'].unique().tolist())}"
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fold_epochs: dict[int, dict[str, int]] = {}
    candidate_epochs: dict[int, int] = {}
    for seed in SEQUENCE_EPOCHS:
        seed_args = training_args(args, seed)
        fold_epochs[seed] = {}
        for held_out_site in SITES:
            outer_training = frame.loc[~frame["SiteID"].eq(held_out_site)].copy()
            inner_training, inner_validation = deterministic_inner_split(
                outer_training, args.inner_validation_fraction, seed
            )
            selection_dir = (
                args.output_dir
                / "selection"
                / f"seed_{seed}"
                / f"held_{held_out_site}"
            )
            select_epoch(
                inner_training.reset_index(drop=True),
                inner_validation.reset_index(drop=True),
                seed_args,
                device,
                selection_dir,
            )
            history = json.loads((selection_dir / "inner_history.json").read_text())
            fold_epochs[seed][held_out_site] = robust_epoch_from_history(
                history, SEQUENCE_SELECTION_NEAR_BEST_AC
            )
        candidate_epochs[seed] = median_epoch(list(fold_epochs[seed].values()))

    predictions: dict[str, dict[str, dict[int, np.ndarray]]] = {
        "baseline": {site: {} for site in SITES},
        "candidate": {site: {} for site in SITES},
    }
    prediction_rows: list[pd.DataFrame] = []
    for seed, baseline_epoch in SEQUENCE_EPOCHS.items():
        seed_args = training_args(args, seed)
        candidate_epoch = candidate_epochs[seed]
        for held_out_site in SITES:
            outer_training = frame.loc[~frame["SiteID"].eq(held_out_site)].copy()
            held_out = frame.loc[frame["SiteID"].eq(held_out_site)].copy()
            epoch_scores: dict[int, np.ndarray] = {}
            for epoch in sorted({int(baseline_epoch), int(candidate_epoch)}):
                refit_dir = (
                    args.output_dir
                    / "refit"
                    / f"seed_{seed}"
                    / f"held_{held_out_site}"
                    / f"epoch_{epoch}"
                )
                epoch_scores[epoch] = refit_and_predict(
                    outer_training.reset_index(drop=True),
                    held_out.reset_index(drop=True),
                    epoch,
                    seed_args,
                    device,
                    refit_dir,
                )
            predictions["baseline"][held_out_site][seed] = epoch_scores[
                int(baseline_epoch)
            ]
            predictions["candidate"][held_out_site][seed] = epoch_scores[
                int(candidate_epoch)
            ]

    metrics: dict[str, dict[str, dict[str, float | int]]] = {
        "baseline": {},
        "candidate": {},
    }
    for protocol in ("baseline", "candidate"):
        for held_out_site in SITES:
            held_out = frame.loc[frame["SiteID"].eq(held_out_site)].copy()
            members = [
                predictions[protocol][held_out_site][seed]
                for seed in SEQUENCE_EPOCHS
            ]
            ensemble = logit_mean_probabilities(members)
            metrics[protocol][held_out_site] = evaluate_scores(held_out, ensemble)
            rows = pd.DataFrame(
                {
                    "record_id": held_out["record_id"].astype(str).to_numpy(),
                    "SiteID": held_out_site,
                    "label": held_out["_label"].to_numpy(dtype=int),
                    "Age": held_out["_age"].to_numpy(dtype=float),
                    "protocol": protocol,
                    "probability": ensemble,
                }
            )
            for index, seed in enumerate(SEQUENCE_EPOCHS):
                rows[f"seed_{seed}_probability"] = members[index]
            prediction_rows.append(rows)

    gate = selection_gate(metrics["baseline"], metrics["candidate"])
    selected_epochs = candidate_epochs if gate["passed"] else dict(SEQUENCE_EPOCHS)
    summary = {
        "status": "complete",
        "protocol": "v19_fresh_e1_adaptive_sequence_epoch_gate_v1",
        "fallback_epochs": {str(key): int(value) for key, value in SEQUENCE_EPOCHS.items()},
        "candidate_epochs": {str(key): int(value) for key, value in candidate_epochs.items()},
        "selected_epochs": {str(key): int(value) for key, value in selected_epochs.items()},
        "fold_selected_epochs": {
            str(seed): {site: int(epoch) for site, epoch in values.items()}
            for seed, values in fold_epochs.items()
        },
        "selection": {
            "maximum_epochs": int(args.max_epochs),
            "inner_validation_fraction": float(args.inner_validation_fraction),
            "near_best_ac_tolerance": SEQUENCE_SELECTION_NEAR_BEST_AC,
            "tie_break": "earliest epoch within tolerance, then median across LOSO folds",
            "metric": "age_conditioned_auroc",
        },
        "metrics": metrics,
        "gate": gate,
        "binary_threshold": 0.5,
        "binary_threshold_reason": "V18 reward-leading operating point retained",
    }
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.output_dir / "oof_predictions.csv.gz", index=False
    )
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # The parent pipeline records this failure and safely restores V18 budgets.
        raise SystemExit(f"adaptive epoch selection failed: {type(error).__name__}: {error}")
