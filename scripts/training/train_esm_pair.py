#!/usr/bin/env python3
"""Train a partner-conditioned, residue-level chain-pair model."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.utils.data import BatchSampler, DataLoader

from ppcbind.data.esm_pair_dataset import ESMPairSiteDataset, collate_esm_pair_features
from ppcbind.data.rcsb_pair_dataset import (
    RCSB_NO_GRAPH_ABLATION,
    RCSB_PAIR_FEATURE_DIM,
    RCSBPairDataset,
)
from ppcbind.models.esm_pair import ESMPairSiteClassifier
from ppcbind.pair_checkpoint import (
    atomic_torch_save, capture_rng_state, require_matching_config, restore_rng_state,
)


MODEL_KEYS = (
    "esm_embeddings",
    "seq_features",
    "protein_mask",
    "chain_ids",
    "chain_rel_pos",
    "protein_rel_pos",
    "prottrans_embeddings",
    "contact_edge_index",
    "contact_edge_scores",
    "aux_contact_edge_index",
    "aux_contact_edge_scores",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairTokenBatchSampler(BatchSampler):
    def __init__(
        self,
        lengths: list[tuple[int, int]],
        batch_size: int,
        max_tokens: int,
        max_pair_cells: int,
        max_single_pair_cells: int,
        shuffle: bool,
        seed: int,
        bucket_size: int = 512,
        joint_grid: bool = False,
    ) -> None:
        self.lengths = [(int(left), int(right)) for left, right in lengths]
        self.batch_size = max(1, int(batch_size))
        self.max_tokens = max(1, int(max_tokens))
        self.max_pair_cells = max(1, int(max_pair_cells))
        self.max_single_pair_cells = max(self.max_pair_cells, int(max_single_pair_cells))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size, int(bucket_size))
        self.epoch = 0
        self.start_batch = 0
        self.joint_grid = joint_grid
        oversized = [
            (idx, left, right)
            for idx, (left, right) in enumerate(self.lengths)
            if self._cells(left, right) > self.max_single_pair_cells
        ]
        if oversized:
            raise ValueError(
                f"{len(oversized)} pairs exceed max_single_pair_cells={self.max_single_pair_cells}; "
                f"first={oversized[:5]}"
            )

    def _cells(self, left: int, right: int) -> int:
        return (left + right) ** 2 if self.joint_grid else left * right

    def _pack(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        max_left = 0
        max_right = 0
        pair_cells = 0
        for idx in indices:
            left, right = self.lengths[idx]
            proposed_left = max(max_left, left)
            proposed_right = max(max_right, right)
            padded_tokens = (proposed_left + proposed_right) * (len(current) + 1)
            proposed_pair_cells = pair_cells + self._cells(left, right)
            if current and (
                len(current) >= self.batch_size
                or padded_tokens > self.max_tokens
                or proposed_pair_cells > self.max_pair_cells
            ):
                batches.append(current)
                current = []
                max_left = 0
                max_right = 0
                pair_cells = 0
            current.append(idx)
            max_left = max(max_left, left)
            max_right = max(max_right, right)
            pair_cells += self._cells(left, right)
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        indices = sorted(range(len(self.lengths)), key=lambda idx: sum(self.lengths[idx]))
        batches = self._pack(indices)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        self.epoch += 1
        if not 0 <= self.start_batch <= len(batches):
            raise ValueError("Resume batch cursor is outside the epoch")
        yield from batches[self.start_batch:]

    def __len__(self) -> int:
        indices = sorted(range(len(self.lengths)), key=lambda idx: sum(self.lengths[idx]))
        return len(self._pack(indices))


class PairResidueLoss(nn.Module):
    def __init__(self, pos_weight: float, label_smoothing: float, ratio_weight: float) -> None:
        super().__init__()
        self.register_buffer("class_weight", torch.tensor([1.0, float(pos_weight)], dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)
        self.ratio_weight = float(ratio_weight)

    def _one_side(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != -100
        per_chain: list[torch.Tensor] = []
        for idx in range(labels.shape[0]):
            keep = valid[idx]
            if not bool(keep.any()):
                continue
            residue_loss = torch.nn.functional.cross_entropy(
                logits[idx][keep].float(),
                labels[idx][keep],
                weight=self.class_weight,
                label_smoothing=self.label_smoothing,
                reduction="mean",
            )
            if self.ratio_weight > 0:
                predicted_ratio = torch.softmax(logits[idx][keep].float(), dim=-1)[:, 1].mean()
                observed_ratio = labels[idx][keep].float().mean()
                residue_loss = residue_loss + self.ratio_weight * (predicted_ratio - observed_ratio).square()
            per_chain.append(residue_loss)
        if not per_chain:
            return logits.sum() * 0.0
        return torch.stack(per_chain).mean()

    def forward(
        self,
        left_logits: torch.Tensor,
        left_labels: torch.Tensor,
        right_logits: torch.Tensor,
        right_labels: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            self._one_side(left_logits, left_labels) + self._one_side(right_logits, right_labels)
        )


def _pair_contact_loss(
    pair_logits: list[torch.Tensor],
    contact_indices: list[torch.Tensor],
    left_observed_masks: list[torch.Tensor],
    right_observed_masks: list[torch.Tensor],
    hard_negative_fraction: float = 0.0,
    negatives_per_positive: int = 20,
    max_sampled_negatives: int = 65536,
) -> torch.Tensor:
    if not pair_logits:
        if contact_indices:
            return contact_indices[0].new_zeros((), dtype=torch.float32)
        return torch.zeros((), dtype=torch.float32)
    losses: list[torch.Tensor] = []
    for logits, contact_index, left_observed, right_observed in zip(
        pair_logits, contact_indices, left_observed_masks, right_observed_masks
    ):
        contact_index = contact_index.to(logits.device, non_blocking=True)
        left_observed = left_observed.to(logits.device, non_blocking=True)
        right_observed = right_observed.to(logits.device, non_blocking=True)
        valid = left_observed[:, None] & right_observed[None, :]
        positive_mask = torch.zeros_like(valid)
        if contact_index.numel():
            positive_mask[contact_index[0], contact_index[1]] = True
        positive_mask &= valid
        negative_mask = valid & ~positive_mask
        if not bool(positive_mask.any()) or not bool(negative_mask.any()):
            continue
        positive_loss = torch.nn.functional.softplus(-logits.float()[positive_mask]).mean()
        negative_values = logits.float()[negative_mask]
        if hard_negative_fraction > 0:
            count = min(negative_values.numel(), max_sampled_negatives,
                        max(1, contact_index.shape[1] * negatives_per_positive))
            hard_values = torch.topk(negative_values, count).values
            random_indices = torch.randint(negative_values.numel(), (count,), device=logits.device)
            random_values = negative_values[random_indices]
            negative_loss = (
                hard_negative_fraction * torch.nn.functional.softplus(hard_values).mean()
                + (1.0 - hard_negative_fraction) * torch.nn.functional.softplus(random_values).mean()
            )
        else:
            negative_loss = torch.nn.functional.softplus(negative_values).mean()
        losses.append(0.5 * (positive_loss + negative_loss))
    if not losses:
        return pair_logits[0].sum() * 0.0
    return torch.stack(losses).mean()


def _site_contact_consistency(output: dict[str, Any], batch: dict[str, Any], topk: int = 4) -> torch.Tensor:
    """Weak agreement between site probability and top-k contact evidence.

    This is an auxiliary surrogate, not a calibrated probability of any contact.
    Only observed residue pairs enter either directional reduction.
    """
    losses = []
    for idx, contact_logits in enumerate(output["pair_contact_logits"]):
        masks = [batch[f"{side}_observed_mask"][idx].to(contact_logits.device)
                 for side in ("left", "right")]
        for side, scores, query_mask, partner_mask in (
            ("left", contact_logits, masks[0], masks[1]),
            ("right", contact_logits.T, masks[1], masks[0]),
        ):
            valid_scores = scores[query_mask][:, partner_mask]
            if not valid_scores.numel():
                continue
            evidence = valid_scores.float().topk(min(topk, valid_scores.shape[1]), dim=1).values.sigmoid().mean(1)
            site_logits = output[f"{side}_logits"][idx, :scores.shape[0]][query_mask]
            site_probability = site_logits.float().softmax(-1)[:, 1]
            losses.append((site_probability - evidence).square().mean())
    return torch.stack(losses).mean() if losses else output["left_logits"].sum() * 0.0


def _base_dataset_kwargs(data: dict[str, Any], training: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "esm_root": data["esm_root"],
        "sequence_feature_root": data.get("sequence_feature_root"),
        "primary_embedding_root": data.get("primary_embedding_root"),
        "prottrans_embedding_root": data.get("prottrans_embedding_root"),
        "contact_graph_root": data.get("contact_graph_root"),
        "aux_contact_graph_root": data.get("aux_contact_graph_root"),
        "require_sequence_features": bool(data.get("require_sequence_features", False)),
        "require_primary_embeddings": bool(data.get("require_primary_embeddings", False)),
        "require_prottrans_embeddings": bool(data.get("require_prottrans_embeddings", False)),
        "seed": seed,
        "preload": bool(training.get("preload", False)),
        "strict_ids": True,
        "strict_label_metadata": False,
        "strict_sequence_feature_metadata": bool(training.get("strict_sequence_feature_metadata", False)),
        "require_contact_graph": bool(data.get("require_contact_graph", False)),
        "require_aux_contact_graph": bool(data.get("require_aux_contact_graph", False)),
        "chain_filter_manifest": data["chain_filter_manifest"],
        "payload_cache_size": int(training.get("payload_cache_size", 8)),
    }


def _build_loader(
    config: dict[str, Any], split: str, shuffle: bool
) -> tuple[ESMPairSiteDataset | RCSBPairDataset, DataLoader]:
    data = config["data"]
    training = config["training"]
    seed = int(training.get("seed", 42))
    backend = data.get("pair_data_backend", "legacy")
    if backend == "rcsb_observed":
        smoke_keys = ("smoke_max_chain_length", "max_train_pairs", "max_val_pairs")
        if split == "test" and any(training.get(key) is not None for key in smoke_keys):
            raise ValueError("RCSB test loader cannot be built with smoke limits configured")
        model_config = config.get("model", {})
        if model_config.get("d_seq") != RCSB_PAIR_FEATURE_DIM:
            raise ValueError("RCSB_S5_noESM2contact requires model.d_seq=90")
        if model_config.get("use_seq_features", True) is not True:
            raise ValueError("RCSB_S5_noESM2contact requires model.use_seq_features=true")
        if model_config.get("use_contact_graph") is not False:
            raise ValueError("RCSB_S5_noESM2contact requires model.use_contact_graph=false")
        if data.get("contact_graph_mode") != RCSB_NO_GRAPH_ABLATION:
            raise ValueError(
                f"RCSB loader requires data.contact_graph_mode={RCSB_NO_GRAPH_ABLATION!r}"
            )
        dataset = RCSBPairDataset(
            pair_manifest=data["pair_manifest"],
            pair_label_root=data["pair_label_root"],
            split=split,
            esm_feature_root=data["rcsb_esm_feature_root"],
            raw_ps_root=data["rcsb_raw_ps_root"],
            normalization_stats=data["rcsb_normalization_stats"],
            feature_mode=data.get("rcsb_feature_mode", ""),
            contact_graph_mode=data["contact_graph_mode"],
            feature_cache_size=data.get("rcsb_feature_cache_size", 64),
        )
        length_limit_value = training.get("smoke_max_chain_length")
        cap_key = {"train": "max_train_pairs", "val": "max_val_pairs"}.get(split)
        pair_cap_value = training.get(cap_key) if cap_key is not None else None
        if length_limit_value is not None or pair_cap_value is not None:
            def _positive_limit(value: Any, key: str) -> int:
                if isinstance(value, bool):
                    raise ValueError(f"training.{key} must be a positive integer")
                try:
                    limit = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"training.{key} must be a positive integer") from exc
                if limit <= 0 or (isinstance(value, float) and value != limit):
                    raise ValueError(f"training.{key} must be a positive integer")
                return limit

            source_count = len(dataset.rows)
            selected_rows = dataset.rows
            length_limit = (
                _positive_limit(length_limit_value, "smoke_max_chain_length")
                if length_limit_value is not None
                else None
            )
            if length_limit is not None:
                selected_rows = [
                    row
                    for row in selected_rows
                    if int(row["left_length"]) <= length_limit and int(row["right_length"]) <= length_limit
                ]
            pair_cap = _positive_limit(pair_cap_value, cap_key) if pair_cap_value is not None else None
            if pair_cap is not None:
                selected_rows = selected_rows[:pair_cap]
            dataset.rows = selected_rows
            selected_ids = [str(row["pair_id"]) for row in selected_rows]
            print(
                f"RCSB smoke selection split={split} source={source_count} "
                f"max_chain_length={length_limit} pair_cap={pair_cap} "
                f"selected={len(selected_ids)} pair_ids={selected_ids!r}"
            )
            if not selected_rows:
                raise ValueError(f"RCSB smoke selection produced no pairs for split={split!r}")
    elif backend == "legacy":
        dataset = ESMPairSiteDataset(
            pair_manifest=data["pair_manifest"],
            pair_label_root=data["pair_label_root"],
            split=split,
            base_dataset_kwargs=_base_dataset_kwargs(data, training, seed),
        )
    else:
        raise ValueError(f"Unknown pair_data_backend: {backend!r}")
    sampler = PairTokenBatchSampler(
        dataset.pair_lengths,
        batch_size=int(training.get("batch_size" if split == "train" else "eval_batch_size", 16)),
        max_tokens=int(training.get("max_batch_tokens", 32768)),
        max_pair_cells=int(training.get("max_pair_cells", 500000)),
        max_single_pair_cells=int(training.get("max_single_pair_cells", 6000000)),
        shuffle=shuffle,
        seed=seed,
        bucket_size=int(training.get("length_bucket_size", 512)),
        joint_grid=config.get("pair_model", {}).get("mode") == "pairformer",
    )
    workers = int(training.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)) and torch.cuda.is_available(),
        "collate_fn": collate_esm_pair_features,
    }
    if int(training.get("checkpoint_every_steps", 0)) > 0:
        # Worker startup must not perturb model dropout RNG after resuming.
        kwargs["generator"] = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[split])
    if workers > 0:
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
    return dataset, DataLoader(dataset, **kwargs)


def _move_side(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in MODEL_KEYS and isinstance(value, torch.Tensor)
    }
    return inputs, batch["labels"].to(device, non_blocking=True)


def _mcc(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.unique(target).size < 2 or np.unique(prediction).size < 2:
        return 0.0
    return float(matthews_corrcoef(target, prediction))


def _threshold_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score >= threshold
    tp = int(np.sum(prediction & (target == 1)))
    fp = int(np.sum(prediction & (target == 0)))
    fn = int(np.sum(~prediction & (target == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"threshold": float(threshold), "f1": float(f1), "mcc": _mcc(target, prediction.astype(np.int64))}


def _summarize_predictions(
    chains: list[tuple[str, np.ndarray, np.ndarray]],
    threshold_grid: list[float],
    allow_threshold_search: bool = True,
) -> dict[str, Any]:
    target = np.concatenate([row[2] for row in chains])
    score = np.concatenate([row[1] for row in chains])
    pooled_ap = float(average_precision_score(target, score)) if np.any(target == 1) else 0.0
    pooled_auroc = float(roc_auc_score(target, score)) if np.unique(target).size == 2 else 0.0
    chain_ap = [float(average_precision_score(y, p)) if np.any(y == 1) else 0.0 for _, p, y in chains]
    at_half = _threshold_metrics(target, score, 0.5)
    summary = {
        "n_directional_chains": len(chains),
        "n_residues": int(target.size),
        "positive_fraction": float(target.mean()),
        "pr_auc": pooled_ap,
        "auroc": pooled_auroc,
        "chain_ap_macro": float(np.mean(chain_ap)),
        "chain_ap_median": float(np.median(chain_ap)),
        "f1_at_0p5": at_half["f1"],
        "mcc_at_0p5": at_half["mcc"],
    }
    if allow_threshold_search:
        grid = [_threshold_metrics(target, score, threshold) for threshold in threshold_grid]
        best_f1 = max(grid, key=lambda row: (row["f1"], row["mcc"]))
        best_mcc = max(grid, key=lambda row: (row["mcc"], row["f1"]))
        summary.update(
            best_threshold=best_f1["threshold"],
            f1_best_threshold=best_f1["f1"],
            mcc_at_best_f1_threshold=_threshold_metrics(target, score, best_f1["threshold"])["mcc"],
            best_mcc_threshold=best_mcc["threshold"],
            mcc_best_threshold=best_mcc["mcc"],
        )
    return summary


def _metrics_at_frozen_threshold(
    chains: list[tuple[str, np.ndarray, np.ndarray]], threshold: float
) -> dict[str, float]:
    target = np.concatenate([row[2] for row in chains])
    score = np.concatenate([row[1] for row in chains])
    return _threshold_metrics(target, score, threshold)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PairResidueLoss,
    device: torch.device,
    thresholds: list[float],
    amp: bool,
    pair_loss_weight: float,
    allow_threshold_search: bool = True,
) -> tuple[dict[str, Any], list[tuple[str, np.ndarray, np.ndarray]]]:
    model.eval()
    losses: list[float] = []
    site_losses: list[float] = []
    pair_losses: list[float] = []
    chains: list[tuple[str, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            left_inputs, left_labels = _move_side(batch["left"], device)
            right_inputs, right_labels = _move_side(batch["right"], device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                output = model(left_inputs, right_inputs)
                site_loss = criterion(
                    output["left_logits"], left_labels, output["right_logits"], right_labels
                )
                pair_loss = (
                    _pair_contact_loss(
                        output["pair_contact_logits"],
                        batch["contact_index"],
                        batch["left_observed_mask"],
                        batch["right_observed_mask"],
                    ).to(site_loss.device)
                    if pair_loss_weight > 0
                    else site_loss.new_zeros(())
                )
                loss = site_loss + pair_loss_weight * pair_loss
            losses.append(float(loss.item()))
            site_losses.append(float(site_loss.item()))
            pair_losses.append(float(pair_loss.item()))
            for side, labels, logits in (
                ("left", left_labels, output["left_logits"]),
                ("right", right_labels, output["right_logits"]),
            ):
                probabilities = torch.softmax(logits.float(), dim=-1)[..., 1]
                for idx, pair_id in enumerate(batch["pair_id"]):
                    valid = labels[idx] != -100
                    chains.append(
                        (
                            f"{pair_id}/{side}",
                            probabilities[idx][valid].cpu().numpy(),
                            labels[idx][valid].cpu().numpy().astype(np.int64),
                        )
                    )
    metrics = _summarize_predictions(chains, thresholds, allow_threshold_search=allow_threshold_search)
    metrics["loss"] = float(np.mean(losses))
    metrics["site_loss"] = float(np.mean(site_losses))
    metrics["pair_contact_loss"] = float(np.mean(pair_losses))
    return metrics, chains


def _symmetry_check(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    batch = next(iter(loader))
    left_inputs, _ = _move_side(batch["left"], device)
    right_inputs, _ = _move_side(batch["right"], device)
    with torch.no_grad():
        forward = model(left_inputs, right_inputs)
        reverse = model(right_inputs, left_inputs)
    return max(
        float((forward["left_logits"] - reverse["right_logits"]).abs().max().item()),
        float((forward["right_logits"] - reverse["left_logits"]).abs().max().item()),
    )


def _set_base_trainable(model: ESMPairSiteClassifier, trainable: bool) -> None:
    for parameter in model.base.parameters():
        parameter.requires_grad_(trainable)


def _cosine_scale(step: int, total: int, warmup: int, minimum: float) -> float:
    if warmup > 0 and step <= warmup:
        return minimum + (1.0 - minimum) * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class TrainingInterrupted(Exception):
    """Validation interrupted after training progress was checkpointed."""


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume this run's atomic progress.pt")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    training = config["training"]
    torch.set_num_threads(int(training.get("torch_num_threads", 4)))
    output_dir = Path(training["output_dir"])
    # Atomic reservation prevents two jobs from writing into the same run.
    resume_state = None
    if args.resume:
        resume_state = torch.load(output_dir / "progress.pt", map_location="cpu", weights_only=False)
        require_matching_config(resume_state["training_config"], config)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    stop_requested = False
    evaluating = False
    checkpoint_interval = int(training.get("checkpoint_every_steps", 0))

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        if evaluating and checkpoint_interval:
            raise TrainingInterrupted()

    def evaluate_interruptible(*values: Any, **options: Any) -> Any:
        nonlocal evaluating
        try:
            evaluating = True
            if stop_requested and checkpoint_interval:
                raise TrainingInterrupted()
            return _evaluate(*values, **options)
        finally:
            evaluating = False

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    seed = int(training.get("seed", 42))
    _seed_everything(seed)
    device = torch.device(str(training.get("device", "cuda:0")))
    thresholds = [float(value) for value in np.arange(0.01, 1.0, 0.01)]

    train_dataset, train_loader = _build_loader(config, "train", True)
    val_dataset, val_loader = _build_loader(config, "val", False)
    positive, negative = train_dataset.class_counts
    pair_positive, pair_negative = train_dataset.pair_class_counts
    pos_weight = min(float(training.get("max_pos_weight", 20.0)), negative / max(1, positive))

    model = ESMPairSiteClassifier(config["model"], config.get("pair_model")).to(device)
    initialization = str(training.get("initialization", "warm_start"))
    checkpoint_path = None
    if initialization == "scratch":
        if training.get("init_single_chain_checkpoint") or int(training.get("freeze_base_epochs", 0)):
            raise ValueError("Scratch initialization must have no checkpoint and no frozen-base epochs")
    elif initialization == "warm_start":
        checkpoint_path = Path(training["init_single_chain_checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_base_checkpoint(checkpoint)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    freeze_base_epochs = int(training.get("freeze_base_epochs", 0 if initialization == "scratch" else 1))
    _set_base_trainable(model, freeze_base_epochs <= 0)
    criterion = PairResidueLoss(
        pos_weight=pos_weight,
        label_smoothing=float(training.get("label_smoothing", 0.05)),
        ratio_weight=float(training.get("ratio_loss_weight", 0.25)),
    ).to(device)
    pair_loss_weight = float(training.get("pair_contact_loss_weight", 0.0))
    hard_fraction = float(training.get("hard_negative_fraction", 0.0))
    consistency_weight = float(training.get("site_contact_consistency_weight", 0.0))
    if not 0 <= hard_fraction <= 1 or consistency_weight < 0:
        raise ValueError("Invalid hard-negative fraction or consistency weight")
    if hard_fraction and pair_loss_weight <= 0:
        raise ValueError("Hard-negative sampling requires contact supervision")
    if (pair_loss_weight > 0 or consistency_weight > 0) and model.mode != "outer_product":
        raise ValueError("Pair auxiliary losses require outer-product mode")
    base_parameters = list(model.base.parameters())
    pair_parameters = list(model.adapter.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": float(training.get("base_lr", 2e-5))},
            {"params": pair_parameters, "lr": float(training.get("pair_lr", 1e-4))},
        ],
        weight_decay=float(training.get("weight_decay", 0.1)),
        eps=float(training.get("adam_eps", 1e-7)),
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    amp = bool(training.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    epochs = int(training.get("epochs", 15))
    batches_per_epoch = len(train_loader)
    total_steps = epochs * batches_per_epoch
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.05)))
    min_lr_scale = float(training.get("min_lr_scale", 0.02))
    global_step = 0
    run_contract = {
        "config": config,
        "pair_task": "known pair -> bidirectional partner-specific residue labels",
        "selection": "validation chain_ap_macro; test evaluated once after checkpoint lock",
        "n_train_pairs": len(train_dataset),
        "n_val_pairs": len(val_dataset),
        "train_batches_per_epoch": batches_per_epoch,
        "n_test_pairs": "not_read_until_checkpoint_lock",
        "train_positive": positive,
        "train_negative": negative,
        "train_pair_positive": pair_positive,
        "train_pair_negative": pair_negative,
        "pair_contact_loss_weight": pair_loss_weight,
        "pos_weight": pos_weight,
        "initialization": initialization,
        "init_single_chain_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    (output_dir / "config.json").write_text(json.dumps(run_contract, indent=2, default=str) + "\n")
    log_path = output_dir / "metrics.jsonl"
    best_metric = -math.inf
    best_epoch = 0
    start_epoch = 1
    resume_batch = 0
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_every_steps cannot be negative")
    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        global_step = int(resume_state["global_step"])
        best_metric = float(resume_state["best_metric"])
        best_epoch = int(resume_state["best_epoch"])
        start_epoch = int(resume_state["epoch"])
        resume_batch = int(resume_state["completed_batches"])
        restore_rng_state(resume_state["rng"])
        print(json.dumps({"event": "resume", "epoch": start_epoch,
                          "completed_batches": resume_batch, "global_step": global_step}), flush=True)

    def save_progress(epoch: int, completed_batches: int, histories: dict[str, list[float]]) -> None:
        atomic_torch_save({
            "training_config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "global_step": global_step, "epoch": epoch,
            "completed_batches": completed_batches, "best_metric": best_metric,
            "best_epoch": best_epoch, "rng": capture_rng_state(), "histories": histories,
        }, output_dir / "progress.pt")

    if resume_state is None and checkpoint_interval:
        save_progress(1, 0, {})
    print(json.dumps({"event": "start", **run_contract}, default=str), flush=True)

    for epoch in range(start_epoch, epochs + 1):
        _set_base_trainable(model, epoch > freeze_base_epochs)
        model.train()
        histories = resume_state.get("histories", {}) if resume_state is not None and epoch == start_epoch else {}
        losses: list[float] = histories.get("loss", [])
        site_losses: list[float] = histories.get("site", [])
        pair_losses: list[float] = histories.get("pair", [])
        consistency_losses: list[float] = histories.get("consistency", [])
        histories = {"loss": losses, "site": site_losses, "pair": pair_losses, "consistency": consistency_losses}
        batch_start = resume_batch if epoch == start_epoch else 0
        train_loader.batch_sampler.epoch = epoch - 1
        train_loader.batch_sampler.start_batch = batch_start
        if stop_requested:
            save_progress(epoch, batch_start, histories)
            return 75
        started = time.time()
        pairs_this_segment = 0
        for step, batch in enumerate(train_loader, batch_start + 1):
            global_step += 1
            scale = _cosine_scale(global_step, total_steps, warmup_steps, min_lr_scale)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * scale
            left_inputs, left_labels = _move_side(batch["left"], device)
            right_inputs, right_labels = _move_side(batch["right"], device)
            pairs_this_segment += int(left_labels.shape[0])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                output = model(left_inputs, right_inputs)
                site_loss = criterion(
                    output["left_logits"], left_labels, output["right_logits"], right_labels
                )
                pair_loss = (
                    _pair_contact_loss(
                        output["pair_contact_logits"],
                        batch["contact_index"],
                        batch["left_observed_mask"],
                        batch["right_observed_mask"],
                        hard_negative_fraction=hard_fraction,
                    ).to(site_loss.device)
                    if pair_loss_weight > 0
                    else site_loss.new_zeros(())
                )
                loss = site_loss + pair_loss_weight * pair_loss
                consistency_loss = (
                    _site_contact_consistency(output, batch)
                    if consistency_weight > 0 else site_loss.new_zeros(())
                )
                loss = loss + consistency_weight * consistency_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            site_losses.append(float(site_loss.item()))
            pair_losses.append(float(pair_loss.item()))
            consistency_losses.append(float(consistency_loss.item()))
            if step % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step,
                            "epoch_batches": batches_per_epoch,
                            "elapsed_segment_seconds": time.time() - started,
                            "batch_start": batch_start,
                            "pairs_per_second_segment": pairs_this_segment / max(time.time() - started, 1e-9),
                            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                            "loss": float(np.mean(losses[-25:])),
                            "site_loss": float(np.mean(site_losses[-25:])),
                            "pair_contact_loss": float(np.mean(pair_losses[-25:])),
                            "lr_base": optimizer.param_groups[0]["lr"],
                            "lr_pair": optimizer.param_groups[1]["lr"],
                        }
                    ),
                    flush=True,
                )
            if stop_requested or (checkpoint_interval and global_step % checkpoint_interval == 0):
                save_progress(epoch, step, histories)
                print(json.dumps({"event": "progress_saved", "epoch": epoch, "step": step,
                                  "global_step": global_step, "stop_requested": stop_requested}), flush=True)
            if stop_requested:
                return 75
        if checkpoint_interval:
            save_progress(epoch, batches_per_epoch, histories)
        val_metrics, _ = evaluate_interruptible(
            model,
            val_loader,
            criterion,
            device,
            thresholds,
            amp,
            pair_loss_weight,
            allow_threshold_search=bool(training.get("search_val_threshold_each_epoch", True)),
        )
        record = {
            "event": "epoch",
            "epoch": epoch,
            "seconds": time.time() - started,
            "train_loss": float(np.mean(losses)),
            "train_site_loss": float(np.mean(site_losses)),
            "train_pair_contact_loss": float(np.mean(pair_losses)),
            "train_consistency_loss": float(np.mean(consistency_losses)),
            "val": val_metrics,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": run_contract,
            "val": val_metrics,
        }
        atomic_torch_save(state, output_dir / "last.pt")
        if float(val_metrics["chain_ap_macro"]) > best_metric:
            best_metric = float(val_metrics["chain_ap_macro"])
            best_epoch = epoch
            atomic_torch_save(state, output_dir / "best.pt")
        if checkpoint_interval or stop_requested:
            save_progress(epoch + 1, 0, {})
        resume_state = None
        if stop_requested:
            return 75

    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    val_metrics, _ = evaluate_interruptible(
        model,
        val_loader,
        criterion,
        device,
        thresholds,
        amp,
        pair_loss_weight,
    )
    if not bool(training.get("evaluate_test", True)):
        symmetry_error = _symmetry_check(model, val_loader, device)
        final = {
            "event": "final",
            "best_epoch": best_epoch,
            "selection_metric": "chain_ap_macro",
            "best_val_chain_ap_macro": best_metric,
            "val": val_metrics,
            "test": None,
            "test_status": "not_evaluated_during_variant_selection",
            "n_test_pairs": "not_read",
            "swap_symmetry_max_abs_logit_error": symmetry_error,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(final) + "\n")
        (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
        print(json.dumps(final), flush=True)
        return 0
    # Constructing the test dataset reads its pair labels/counts, so delay it
    # until the validation-selected checkpoint has been locked and reloaded.
    test_dataset, test_loader = _build_loader(config, "test", False)
    test_metrics, test_chains = evaluate_interruptible(
        model,
        test_loader,
        criterion,
        device,
        thresholds,
        amp,
        pair_loss_weight,
        allow_threshold_search=False,
    )
    frozen = _metrics_at_frozen_threshold(test_chains, float(val_metrics["best_threshold"]))
    symmetry_error = _symmetry_check(model, val_loader, device)
    final = {
        "event": "final",
        "best_epoch": best_epoch,
        "selection_metric": "chain_ap_macro",
        "best_val_chain_ap_macro": best_metric,
        "val": val_metrics,
        "test": test_metrics,
        "test_at_val_threshold": frozen,
        "n_test_pairs": len(test_dataset),
        "swap_symmetry_max_abs_logit_error": symmetry_error,
    }
    with log_path.open("a") as handle:
        handle.write(json.dumps(final) + "\n")
    (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final), flush=True)
    return 0


def main() -> int:
    try:
        return _main()
    except TrainingInterrupted:
        print(json.dumps({"event": "validation_interrupted", "resume_from": "progress.pt"}), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
