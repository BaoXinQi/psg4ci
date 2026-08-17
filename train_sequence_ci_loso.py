#!/usr/bin/env python3
"""Train fixed Raw-only and D+Raw full-night Transformers with site-wise LOSO."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from sequence_ci_model import (
    FullNightEmbeddingDataset,
    LocalFullNightTransformer,
    NaturalWithSiteBalancedBatchSampler,
    collate_full_nights,
)


SEED = 20260806
AGE_GAP = 2.0
MODELS = ("raw_sequence", "demographics_raw_sequence")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def age_conditioned_auroc(
    labels: np.ndarray, scores: np.ndarray, ages: np.ndarray, gap: float = AGE_GAP
) -> tuple[float, int]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    age_mask = np.abs(ages[positives, None] - ages[None, negatives]) <= gap
    pairs = int(age_mask.sum())
    if pairs == 0:
        return float("nan"), 0
    concordance = (
        (scores[positives, None] > scores[None, negatives]).astype(float)
        + 0.5 * (scores[positives, None] == scores[None, negatives]).astype(float)
    )
    return float(concordance[age_mask].mean()), pairs


def evaluate_scores(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float | int]:
    labels = frame["_label"].to_numpy(dtype=int)
    ages = frame["_age"].to_numpy(dtype=float)
    ac, pairs = age_conditioned_auroc(labels, scores, ages)
    auroc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else math.nan
    return {
        "age_conditioned_auroc": ac,
        "age_pair_count": pairs,
        "auroc": auroc,
        "records": int(len(frame)),
        "positives": int(labels.sum()),
    }


def checkpoint_selection_score(
    metrics: Mapping[str, float | int], selection_metric: str
) -> float:
    age_conditioned = float(metrics["age_conditioned_auroc"])
    if selection_metric == "age_conditioned_auroc":
        return age_conditioned if np.isfinite(age_conditioned) else -math.inf
    if selection_metric == "composite":
        values = [age_conditioned, float(metrics["auroc"])]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else -math.inf
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


class DemographicTransform:
    numeric_columns = ("Age", "BMI")
    categorical_columns = ("Sex", "Race", "Ethnicity")

    def __init__(self) -> None:
        self.numeric: list[str] = []
        self.numeric_median = np.empty(0)
        self.numeric_mean = np.empty(0)
        self.numeric_std = np.empty(0)
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "DemographicTransform":
        self.numeric = [column for column in self.numeric_columns if column in frame]
        values = frame[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.numeric_median = np.nanmedian(values, axis=0)
        filled = np.where(np.isfinite(values), values, self.numeric_median)
        self.numeric_mean = filled.mean(axis=0)
        self.numeric_std = np.maximum(filled.std(axis=0), 1e-6)
        for column in self.categorical_columns:
            if column in frame:
                values_string = frame[column].fillna("__MISSING__").astype(str)
                self.categories[column] = sorted(values_string.unique().tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric:
            values = frame[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            missing = (~np.isfinite(values)).astype(np.float32)
            filled = np.where(np.isfinite(values), values, self.numeric_median)
            parts.extend([((filled - self.numeric_mean) / self.numeric_std).astype(np.float32), missing])
        for column, categories in self.categories.items():
            values = frame[column].fillna("__MISSING__").astype(str).to_numpy()
            parts.append(
                np.column_stack([values == category for category in categories]).astype(np.float32)
            )
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(frame), 1), np.float32)


def deterministic_inner_split(
    outer_training: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []
    for _, group in outer_training.groupby(["SiteID", "_label"], sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        count = min(max(int(round(len(indices) * fraction)), 1), max(len(indices) - 1, 0))
        validation_indices.extend(int(value) for value in indices[:count])
    validation_set = set(validation_indices)
    validation = outer_training.loc[sorted(validation_set)].copy()
    training = outer_training.loc[
        [index for index in outer_training.index if index not in validation_set]
    ].copy()
    return training, validation


def subset_smoke(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0:
        return frame
    groups = list(frame.groupby(["SiteID", "_label"], sort=True))
    per_group = max(4, math.ceil(count / len(groups)))
    selected = [
        group.sample(
            n=min(per_group, len(group)),
            random_state=seed + position * 101,
        )
        for position, (_, group) in enumerate(groups)
    ]
    return pd.concat(selected).sort_values(["SiteID", "record_id"]).reset_index(drop=True)


def make_loader(
    frame: pd.DataFrame,
    demographics: np.ndarray,
    batch_size: int,
    workers: int,
    seed: int,
    training: bool,
) -> tuple[DataLoader, NaturalWithSiteBalancedBatchSampler | None]:
    dataset = FullNightEmbeddingDataset(frame, demographics)
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": collate_full_nights,
    }
    if workers:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    if training:
        sampler = NaturalWithSiteBalancedBatchSampler(
            frame["SiteID"].astype(str).tolist(), batch_size, seed
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs), sampler
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs), None


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ages: torch.Tensor,
    site_ids: Sequence[str],
    positive_weight: float,
    pairwise_weight: float,
    pair_scope: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, int]]:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
    )
    positive = labels > 0.5
    negative = ~positive
    pair_counts: dict[str, int] = {}
    if torch.any(positive) and torch.any(negative):
        differences = logits[positive, None] - logits[None, negative]
        age_mask = torch.abs(ages[positive, None] - ages[None, negative]) <= AGE_GAP
        sites = np.asarray(site_ids, dtype=str)
        positive_sites = sites[positive.detach().cpu().numpy()]
        negative_sites = sites[negative.detach().cpu().numpy()]
        age_mask_cpu = age_mask.detach().cpu().numpy()
        for site in sorted(set(sites)):
            site_pairs = (
                (positive_sites[:, None] == site)
                & (negative_sites[None, :] == site)
                & age_mask_cpu
            )
            pair_counts[f"site_{site}"] = int(site_pairs.sum())
        pair_counts["cross_site"] = int(
            (age_mask_cpu & (positive_sites[:, None] != negative_sites[None, :])).sum()
        )
        valid_mask = age_mask
        if pair_scope == "same_site":
            site_mask = torch.as_tensor(
                positive_sites[:, None] == negative_sites[None, :],
                dtype=torch.bool,
                device=logits.device,
            )
            valid_mask = valid_mask & site_mask
        valid_pair_count = int(valid_mask.sum().detach().cpu())
        pairwise = (
            F.softplus(-differences[valid_mask]).mean()
            if valid_pair_count > 0
            else logits.sum() * 0.0
        )
    else:
        pairwise = logits.sum() * 0.0
        valid_pair_count = 0
    return (
        bce + pairwise_weight * pairwise,
        bce,
        pairwise,
        valid_pair_count,
        pair_counts,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    sampler: NaturalWithSiteBalancedBatchSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    positive_weight: float,
    pairwise_weight: float,
    pair_scope: str,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    sampler.set_epoch(epoch)
    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "pairwise": 0.0,
        "records": 0.0,
        "valid_pairs": 0.0,
        "batches": 0.0,
        "batches_with_pairs": 0.0,
    }
    started = time.time()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["blocks"], batch["window_mask"], batch["demographics"])
        loss, bce, pairwise, valid_pair_count, pair_counts = training_loss(
            logits,
            batch["label"],
            batch["age"],
            batch["site_id"],
            positive_weight,
            pairwise_weight,
            pair_scope,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite sequence loss: {batch['record_id']}")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite sequence gradient")
        optimizer.step()
        count = int(len(batch["label"]))
        totals["loss"] += float(loss.detach()) * count
        totals["bce"] += float(bce.detach()) * count
        totals["pairwise"] += float(pairwise.detach()) * count
        totals["records"] += count
        totals["valid_pairs"] += valid_pair_count
        totals["batches"] += 1
        totals["batches_with_pairs"] += int(valid_pair_count > 0)
        for pair_group, pair_count in pair_counts.items():
            key = f"valid_pairs_{pair_group}"
            totals[key] = totals.get(key, 0.0) + pair_count
    records = max(totals.pop("records"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    batches_with_pairs = totals.pop("batches_with_pairs")
    valid_pairs = totals.pop("valid_pairs")
    pair_group_totals = {
        key: int(value)
        for key, value in list(totals.items())
        if key.startswith("valid_pairs_")
    }
    for key in pair_group_totals:
        totals.pop(key)
    return {
        **{name: value / records for name, value in totals.items()},
        "valid_pair_count": int(valid_pairs),
        "batch_pair_coverage": float(batches_with_pairs / batches),
        **pair_group_totals,
        "elapsed_sec": time.time() - started,
    }


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        logits = model(batch["blocks"], batch["window_mask"], batch["demographics"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite sequence prediction")
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def build_model(
    model_name: str,
    demographic_dimension: int,
    device: torch.device,
    initialization_checkpoint: Path | None = None,
) -> nn.Module:
    model = LocalFullNightTransformer(
        demographic_dimension=demographic_dimension,
        use_demographics=model_name == "demographics_raw_sequence",
        d_model=256,
        dropout=0.15,
    )
    if initialization_checkpoint is not None:
        payload = torch.load(initialization_checkpoint, map_location="cpu")
        state = payload.get("backbone_state")
        if not isinstance(state, Mapping):
            raise RuntimeError(
                f"Missing backbone_state in {initialization_checkpoint}"
            )
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("head.")
            and not key.startswith("demographic_encoder.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Incompatible SSL initialization: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return model.to(device)


def select_epoch(
    model_name: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    demo_transform: DemographicTransform,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    initialization_checkpoint: Path | None,
) -> int:
    demo_train = demo_transform.transform(training)
    demo_validation = demo_transform.transform(validation)
    train_loader, sampler = make_loader(
        training, demo_train, args.batch_size, args.num_workers, args.seed, True
    )
    validation_loader, _ = make_loader(
        validation, demo_validation, args.batch_size, args.num_workers, args.seed, False
    )
    set_seed(args.seed)
    model = build_model(
        model_name,
        demo_train.shape[1],
        device,
        initialization_checkpoint,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    positives = float(training["_label"].sum())
    positive_weight = (len(training) - positives) / max(positives, 1.0)
    best_score = -math.inf
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    assert sampler is not None
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            sampler,
            optimizer,
            device,
            epoch,
            positive_weight,
            args.pairwise_weight,
            args.pair_scope,
            args.gradient_clip,
        )
        scores = predict(model, validation_loader, device)
        metrics = evaluate_scores(validation, scores)
        selection = checkpoint_selection_score(metrics, args.selection_metric)
        history.append({"epoch": epoch, "training": train_metrics, "validation": metrics, "selection": selection})
        print(
            json.dumps(
                {
                    "phase": "inner_selection",
                    "model": model_name,
                    "epoch": epoch,
                    "training_loss": train_metrics["loss"],
                    "training_seconds": train_metrics["elapsed_sec"],
                    "validation_ac": metrics["age_conditioned_auroc"],
                    "validation_auroc": metrics["auroc"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if selection > best_score + 1e-4:
            best_score = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inner_history.json").write_text(json.dumps(history, indent=2) + "\n")
    if best_state is None:
        raise RuntimeError("No finite inner-validation checkpoint")
    torch.save(
        {"model_state": best_state, "selected_epoch": best_epoch, "model_name": model_name},
        output_dir / "inner_best.pt",
    )
    return best_epoch


def refit_and_predict(
    model_name: str,
    training: pd.DataFrame,
    test: pd.DataFrame,
    selected_epoch: int,
    demo_transform: DemographicTransform,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    initialization_checkpoint: Path | None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    demo_train = demo_transform.transform(training)
    demo_test = demo_transform.transform(test)
    train_loader, sampler = make_loader(
        training, demo_train, args.batch_size, args.num_workers, args.seed + 17, True
    )
    test_loader, _ = make_loader(
        test, demo_test, args.batch_size, args.num_workers, args.seed, False
    )
    set_seed(args.seed)
    model = build_model(
        model_name,
        demo_train.shape[1],
        device,
        initialization_checkpoint,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    positives = float(training["_label"].sum())
    positive_weight = (len(training) - positives) / max(positives, 1.0)
    history: list[dict[str, float]] = []
    assert sampler is not None
    for epoch in range(1, selected_epoch + 1):
        epoch_metrics = train_epoch(
                model,
                train_loader,
                sampler,
                optimizer,
                device,
                epoch,
                positive_weight,
                args.pairwise_weight,
                args.pair_scope,
                args.gradient_clip,
            )
        history.append(epoch_metrics)
        print(
            json.dumps(
                {
                    "phase": "outer_refit",
                    "model": model_name,
                    "epoch": epoch,
                    "epochs": selected_epoch,
                    "training_loss": epoch_metrics["loss"],
                    "training_seconds": epoch_metrics["elapsed_sec"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "selected_epoch": selected_epoch,
            "model_name": model_name,
            "demographic_transform": demo_transform.__dict__,
        },
        output_dir / "final_model.pt",
    )
    return predict(model, test_loader, device), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialization-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--held-out-sites", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-weight", type=float, default=0.15)
    parser.add_argument(
        "--pair-scope", choices=("all", "same_site"), default="all"
    )
    parser.add_argument(
        "--selection-metric",
        choices=("composite", "age_conditioned_auroc"),
        default="age_conditioned_auroc",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.15)
    parser.add_argument("--smoke-records", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size < 2 or args.max_epochs < args.minimum_epochs:
        raise ValueError("Invalid batch or epoch configuration")
    frame = pd.read_parquet(args.manifest).copy()
    embeddings = pd.read_parquet(args.embedding_manifest).copy()
    frame["record_id"] = frame["record_id"].astype(str)
    embeddings["record_id"] = embeddings["record_id"].astype(str)
    frame = frame.merge(
        embeddings[["record_id", "shard_path", "group_key", "embedding_count", "complete_epoch_count"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_embedding"),
    )
    if len(frame) != 6600:
        raise RuntimeError(f"Expected 6600 merged records, found {len(frame)}")
    frame["SiteID"] = frame["SiteID"].astype(str)
    label_column = "label" if "label" in frame else "Cognitive_Impairment"
    frame["_label"] = pd.to_numeric(frame[label_column], errors="raise").astype(int)
    frame["_age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame = subset_smoke(frame, args.smoke_records, args.seed)
    sites = args.held_out_sites or sorted(frame["SiteID"].unique())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {
        model_name: np.full(len(frame), np.nan, dtype=np.float64) for model_name in args.models
    }
    fold_rows: list[dict[str, Any]] = []

    for held_out_site in sites:
        outer_test = frame.loc[frame["SiteID"].eq(held_out_site)].copy()
        outer_training = frame.loc[~frame["SiteID"].eq(held_out_site)].copy()
        inner_training, inner_validation = deterministic_inner_split(
            outer_training, args.inner_validation_fraction, args.seed
        )
        initialization_checkpoint = None
        if args.initialization_root is not None:
            initialization_checkpoint = (
                args.initialization_root
                / f"held_{held_out_site}"
                / "ssl_backbone.pt"
            )
            if not initialization_checkpoint.is_file():
                raise FileNotFoundError(initialization_checkpoint)
        for model_name in args.models:
            fold_dir = args.output_dir / model_name / f"held_{held_out_site}"
            prediction_path = fold_dir / "fold_predictions.npz"
            metrics_path = fold_dir / "fold_metrics.json"
            if prediction_path.is_file() and metrics_path.is_file():
                with np.load(prediction_path, allow_pickle=True) as payload:
                    cached_ids = payload["record_id"].astype(str)
                    expected_ids = outer_test["record_id"].astype(str).to_numpy()
                    if not np.array_equal(cached_ids, expected_ids):
                        raise RuntimeError(f"Cached fold order mismatch: {fold_dir}")
                    scores = payload["scores"].astype(np.float64)
                metrics = json.loads(metrics_path.read_text())
                predictions[model_name][outer_test.index.to_numpy()] = scores
                fold_rows.append(metrics)
                print(json.dumps({**metrics, "resumed": True}, sort_keys=True), flush=True)
                continue
            inner_demo = DemographicTransform().fit(inner_training)
            selected_epoch = select_epoch(
                model_name,
                inner_training,
                inner_validation,
                inner_demo,
                args,
                device,
                fold_dir,
                initialization_checkpoint,
            )
            outer_demo = DemographicTransform().fit(outer_training)
            scores, refit_history = refit_and_predict(
                model_name,
                outer_training,
                outer_test,
                selected_epoch,
                outer_demo,
                args,
                device,
                fold_dir,
                initialization_checkpoint,
            )
            positions = outer_test.index.to_numpy()
            predictions[model_name][positions] = scores
            metrics = evaluate_scores(outer_test, scores)
            fold_rows.append(
                {
                    "model": model_name,
                    "held_out_site": held_out_site,
                    "selected_epoch": selected_epoch,
                    **metrics,
                }
            )
            fold_dir.mkdir(parents=True, exist_ok=True)
            prediction_tmp = fold_dir / "fold_predictions.npz.tmp"
            with prediction_tmp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    record_id=outer_test["record_id"].astype(str).to_numpy(),
                    scores=scores,
                )
            prediction_tmp.replace(prediction_path)
            metrics_path.write_text(json.dumps(fold_rows[-1], indent=2, sort_keys=True) + "\n")
            (fold_dir / "refit_history.json").write_text(
                json.dumps(refit_history, indent=2) + "\n"
            )
            print(json.dumps(fold_rows[-1], sort_keys=True), flush=True)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    prediction_frame = frame[["record_id", "patient_id", "SiteID", "_label", "_age"]].copy()
    summary: dict[str, Any] = {
        "protocol": {
            "pair_scope": args.pair_scope,
            "pairwise_weight": float(args.pairwise_weight),
            "selection_metric": args.selection_metric,
            "seed": int(args.seed),
            "initialization_root": (
                str(args.initialization_root)
                if args.initialization_root is not None
                else None
            ),
        },
        "models": {},
    }
    for model_name in args.models:
        prediction_frame[model_name] = predictions[model_name]
        valid = np.isfinite(predictions[model_name])
        model_folds = fold_frame.loc[fold_frame["model"].eq(model_name)]
        pooled = evaluate_scores(frame.loc[valid], predictions[model_name][valid])
        summary["models"][model_name] = {
            "held_out_sites": {
                row.held_out_site: float(row.age_conditioned_auroc)
                for row in model_folds.itertuples(index=False)
            },
            "selected_epochs": {
                row.held_out_site: int(row.selected_epoch)
                for row in model_folds.itertuples(index=False)
            },
            "macro_site_age_conditioned_auroc": float(
                model_folds["age_conditioned_auroc"].mean()
            ),
            "worst_site_age_conditioned_auroc": float(
                model_folds["age_conditioned_auroc"].min()
            ),
            "pooled_oof": pooled,
        }
    prediction_frame.to_parquet(args.output_dir / "loso_predictions.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
