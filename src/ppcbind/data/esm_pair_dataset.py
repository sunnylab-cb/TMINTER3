"""Paired-chain view over the established single-chain M18 feature dataset."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ppcbind.data.esm_site_dataset import ESMProteinSiteDataset, collate_esm_site_features


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def load_pair_manifest(path: str | Path, split: str) -> list[dict[str, Any]]:
    manifest = Path(path)
    split = str(split).lower()
    rows: list[dict[str, Any]] = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "")).lower() != split:
                continue
            rows.append(
                {
                    **row,
                    "left_length": int(row["left_length"]),
                    "right_length": int(row["right_length"]),
                    "left_positive": int(row["left_positive"]),
                    "right_positive": int(row["right_positive"]),
                    "left_observed": int(row.get("left_observed") or 0),
                    "right_observed": int(row.get("right_observed") or 0),
                    "n_contacts": int(row.get("n_contacts") or 0),
                    "n_valid_pair_cells": int(row.get("n_valid_pair_cells") or 0),
                }
            )
    if not rows:
        raise ValueError(f"{manifest}: no rows for split={split!r}")
    return rows


class ESMPairSiteDataset(Dataset):
    """Return two M18 chain payloads and their partner-specific labels."""

    def __init__(
        self,
        pair_manifest: str | Path,
        pair_label_root: str | Path,
        split: str,
        *,
        base_dataset_kwargs: dict[str, Any],
    ) -> None:
        self.pair_manifest = Path(pair_manifest)
        self.pair_label_root = Path(pair_label_root)
        self.split = str(split).lower()
        self.rows = load_pair_manifest(self.pair_manifest, self.split)
        seq_ids = sorted(
            {str(row[side]) for row in self.rows for side in ("left_seq_id", "right_seq_id")}
        )
        kwargs = dict(base_dataset_kwargs)
        kwargs.update(
            ids=seq_ids,
            label_root=None,
            require_labels=False,
            max_residues=None,
            crop_mode="none",
        )
        self.chains = ESMProteinSiteDataset(**kwargs)
        self.chain_index = {str(sample_id): idx for idx, sample_id in enumerate(self.chains.sample_ids)}
        missing = sorted(set(seq_ids) - set(self.chain_index))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} pair chains from base features; first={missing[:10]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        label_path = Path(row["label_path"])
        if not label_path.is_absolute():
            label_path = self.pair_label_root / label_path
        pair_labels = _torch_load(label_path)
        expected_metadata = {
            "pair_id": row["pair_id"],
            "pdb_id": row["pdb_id"],
            "split": self.split,
            "left_seq_id": row["left_seq_id"],
            "right_seq_id": row["right_seq_id"],
        }
        mismatches = {
            key: (str(pair_labels.get(key)), str(expected))
            for key, expected in expected_metadata.items()
            if str(pair_labels.get(key)) != str(expected)
        }
        if mismatches:
            raise ValueError(f"{row['pair_id']}: pair-label metadata mismatch: {mismatches}")
        if float(pair_labels.get("cutoff_angstrom", -1.0)) != 5.5 or pair_labels.get("heavy_only") is not True:
            raise ValueError(f"{row['pair_id']}: expected strict 5.5 A heavy-atom labels")
        left = dict(self.chains[self.chain_index[row["left_seq_id"]]])
        right = dict(self.chains[self.chain_index[row["right_seq_id"]]])
        left_labels = torch.as_tensor(pair_labels["left_labels"], dtype=torch.long)
        right_labels = torch.as_tensor(pair_labels["right_labels"], dtype=torch.long)
        left_observed = torch.as_tensor(pair_labels.get("left_observed_mask"), dtype=torch.bool)
        right_observed = torch.as_tensor(pair_labels.get("right_observed_mask"), dtype=torch.bool)
        contact_index = torch.as_tensor(pair_labels.get("contact_index"), dtype=torch.long)
        if contact_index.ndim != 2 or contact_index.shape[0] != 2:
            raise ValueError(f"{row['pair_id']}: contact_index must have shape [2, E]")
        if left_labels.shape[0] != left["esm_embeddings"].shape[0]:
            raise ValueError(
                f"{row['pair_id']}: left label length {left_labels.shape[0]} != feature length "
                f"{left['esm_embeddings'].shape[0]}"
            )
        if right_labels.shape[0] != right["esm_embeddings"].shape[0]:
            raise ValueError(
                f"{row['pair_id']}: right label length {right_labels.shape[0]} != feature length "
                f"{right['esm_embeddings'].shape[0]}"
            )
        for side, item, observed, labels in (
            ("left", left, left_observed, left_labels),
            ("right", right, right_observed, right_labels),
        ):
            if observed.shape != labels.shape:
                raise ValueError(f"{row['pair_id']}: {side} observed-mask shape mismatch")
            sequence = "".join(str(value).strip().upper() for value in item["residue_names_1"])
            sequence_sha1 = hashlib.sha1(sequence.encode("ascii")).hexdigest()
            stored_sha1 = str(pair_labels.get(f"{side}_sequence_sha1", ""))
            manifest_sha1 = str(row.get(f"{side}_sequence_sha1", ""))
            if not stored_sha1 or sequence_sha1 != stored_sha1 or sequence_sha1 != manifest_sha1:
                raise ValueError(
                    f"{row['pair_id']}: {side} sequence SHA1 mismatch: "
                    f"feature={sequence_sha1} label={stored_sha1} manifest={manifest_sha1}"
                )
            if int(observed.sum().item()) != int(row[f"{side}_observed"]):
                raise ValueError(f"{row['pair_id']}: {side} observed-residue count disagrees")
            if not torch.equal(labels != -100, observed):
                raise ValueError(f"{row['pair_id']}: {side} unknown labels do not match observed mask")
        if int((left_labels == 1).sum().item()) != int(row["left_positive"]):
            raise ValueError(f"{row['pair_id']}: left positive count disagrees with manifest")
        if int((right_labels == 1).sum().item()) != int(row["right_positive"]):
            raise ValueError(f"{row['pair_id']}: right positive count disagrees with manifest")
        if int(contact_index.shape[1]) != int(row["n_contacts"]):
            raise ValueError(f"{row['pair_id']}: contact count disagrees with manifest")
        if int(left_observed.sum() * right_observed.sum()) != int(row["n_valid_pair_cells"]):
            raise ValueError(f"{row['pair_id']}: valid pair-cell count disagrees with manifest")
        if contact_index.numel():
            if int(contact_index[0].min()) < 0 or int(contact_index[0].max()) >= left_labels.shape[0]:
                raise ValueError(f"{row['pair_id']}: left contact index is out of range")
            if int(contact_index[1].min()) < 0 or int(contact_index[1].max()) >= right_labels.shape[0]:
                raise ValueError(f"{row['pair_id']}: right contact index is out of range")
            projected_left = torch.zeros_like(left_labels)
            projected_right = torch.zeros_like(right_labels)
            projected_left[~left_observed] = -100
            projected_right[~right_observed] = -100
            projected_left[torch.unique(contact_index[0])] = 1
            projected_right[torch.unique(contact_index[1])] = 1
            if not torch.equal(projected_left, left_labels) or not torch.equal(projected_right, right_labels):
                raise ValueError(f"{row['pair_id']}: site labels are not the projection of contact_index")
        left["labels"] = left_labels
        right["labels"] = right_labels
        return {
            "pair_id": row["pair_id"],
            "pdb_id": row["pdb_id"],
            "left": left,
            "right": right,
            "contact_index": contact_index,
            "left_observed_mask": left_observed,
            "right_observed_mask": right_observed,
        }

    @property
    def pair_lengths(self) -> list[tuple[int, int]]:
        return [(int(row["left_length"]), int(row["right_length"])) for row in self.rows]

    @property
    def class_counts(self) -> tuple[int, int]:
        positive = sum(int(row["left_positive"]) + int(row["right_positive"]) for row in self.rows)
        total = sum(int(row["left_observed"]) + int(row["right_observed"]) for row in self.rows)
        return positive, total - positive

    @property
    def pair_class_counts(self) -> tuple[int, int]:
        positive = sum(int(row["n_contacts"]) for row in self.rows)
        total = sum(int(row["n_valid_pair_cells"]) for row in self.rows)
        return positive, total - positive


def collate_esm_pair_features(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_id": [item["pair_id"] for item in items],
        "source_pdb_id": [item["pdb_id"] for item in items],
        "left": collate_esm_site_features([item["left"] for item in items]),
        "right": collate_esm_site_features([item["right"] for item in items]),
        "contact_index": [item["contact_index"] for item in items],
        "left_observed_mask": [item["left_observed_mask"] for item in items],
        "right_observed_mask": [item["right_observed_mask"] for item in items],
    }
