"""Native RCSB observed-pair adapter for the existing paired-site trainer.

The initial adapter consumes StarESMC embeddings and normalized 89D raw-PS
spatial scalars plus an availability flag. It intentionally supplies no
residue contact graph; training must opt into the
``RCSB_S5_noESM2contact`` candidate (not an exact S5 reproduction).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ppcbind.data.esm_pair_dataset import load_pair_manifest


RCSB_PAIR_FEATURE_MODE = "spatial_scalar89_train_norm_plus_availability90"
RCSB_PAIR_RAW_SCALAR_DIM = 89
RCSB_PAIR_FEATURE_DIM = 90
RCSB_NO_GRAPH_ABLATION = "RCSB_S5_noESM2contact"
RCSB_LABEL_CONTRACT = "observed-polymer-heavy-atom-5p5-assembly-v2"
RCSB_ESMC_RESOLVED_REVISION = "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c"
_UID_RE = re.compile(r"^useq_([0-9a-f]{64})$")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _sequence(value: Any, context: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if not isinstance(value, str):
        raise ValueError(f"{context}: sequence must be a string")
    sequence = value.strip().upper()
    if not sequence:
        raise ValueError(f"{context}: empty sequence")
    try:
        sequence.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{context}: sequence must be ASCII") from exc
    return sequence


def _indices(value: Any, key: str, pair_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{pair_id}: {key} must be a list of zero-based residue indices")
    indices: list[int] = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{pair_id}: {key} contains a non-integer residue index")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{pair_id}: {key} contains duplicate residue indices")
    return indices


def compute_train_uid_checksum(pair_manifest: str | Path) -> str:
    """Hash sorted unique train UIDs, newline-delimited with a final newline."""
    uids: set[str] = set()
    with Path(pair_manifest).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split", "")).strip().lower() != "train":
                continue
            for side in ("left", "right"):
                uid = str(row.get(f"{side}_seq_id", "")).strip()
                if _UID_RE.fullmatch(uid) is None:
                    raise ValueError(f"train manifest row has invalid {side}_seq_id: {uid!r}")
                uids.add(uid)
    if not uids:
        raise ValueError(f"{pair_manifest}: no train UIDs for normalization stats")
    canonical = "".join(f"{uid}\n" for uid in sorted(uids))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


class RCSBPairDataset(Dataset):
    """Load one requested split of RCSB pairs into the legacy pair-batch schema.

    JSONL rows must additionally include ``left_observed_indices`` and
    ``right_observed_indices``. Counts alone cannot distinguish observed
    negatives from unobserved residues, so the adapter rejects rows without
    these indices rather than manufacturing labels. Embedding and raw-PS
    tensors must be finite. Stats JSON requires
    89-value ``mean``/``std``, ``feature_count=89``, ``fit_split='train'``,
    and ``train_uid_checksum``. The checksum is SHA256 over sorted unique train
    UIDs, each followed by a newline.
    """

    feature_dim = RCSB_PAIR_FEATURE_DIM

    def __init__(
        self,
        pair_manifest: str | Path,
        pair_label_root: str | Path,
        split: str,
        *,
        esm_feature_root: str | Path,
        raw_ps_root: str | Path,
        normalization_stats: str | Path,
        feature_mode: str,
        contact_graph_mode: str,
        feature_cache_size: int = 64,
    ) -> None:
        self.pair_manifest = Path(pair_manifest)
        self.pair_label_root = Path(pair_label_root)
        self.esm_feature_root = Path(esm_feature_root)
        self.raw_ps_root = Path(raw_ps_root)
        self.normalization_stats_path = Path(normalization_stats)
        self.split = str(split).lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported RCSB split: {split!r}")
        if feature_mode != RCSB_PAIR_FEATURE_MODE:
            raise ValueError(
                f"RCSB adapter supports only feature_mode={RCSB_PAIR_FEATURE_MODE!r}; got {feature_mode!r}"
            )
        if contact_graph_mode != RCSB_NO_GRAPH_ABLATION:
            raise ValueError(
                "RCSB has no verified ESM2 contact graph; set "
                f"contact_graph_mode={RCSB_NO_GRAPH_ABLATION!r} for the initial candidate"
            )
        if isinstance(feature_cache_size, bool) or not isinstance(feature_cache_size, int) or feature_cache_size < 0:
            raise ValueError("RCSB feature_cache_size must be a nonnegative integer")
        self.feature_cache_size = feature_cache_size
        self._feature_cache: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()
        self._feature_cache_pid = os.getpid()
        self.scalar_mean, self.scalar_std = self._load_normalization_stats()
        self.rows = load_pair_manifest(self.pair_manifest, self.split)
        self._row_by_pair_id: set[str] = set()
        for row in self.rows:
            pair_id = str(row["pair_id"])
            if pair_id in self._row_by_pair_id:
                raise ValueError(f"{pair_id}: duplicate pair_id in split={self.split!r}")
            self._row_by_pair_id.add(pair_id)
            for key in ("assembly_id", "left_seq_id", "right_seq_id", "left_sequence_sha1", "right_sequence_sha1"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"{pair_id}: manifest field {key!r} is required")
            if not str(row.get("label_path", "")).strip():
                raise ValueError(f"{pair_id}: manifest field 'label_path' is required")
            try:
                offset = int(row["label_offset"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{pair_id}: label_offset must be an uncompressed JSONL byte offset") from exc
            if offset < 0:
                raise ValueError(f"{pair_id}: label_offset cannot be negative")
            for side in ("left", "right"):
                uid = str(row[f"{side}_seq_id"])
                if _UID_RE.fullmatch(uid) is None:
                    raise ValueError(f"{pair_id}: {side}_seq_id is not a useq_SHA256 UID: {uid!r}")

    def _load_normalization_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            stats = json.loads(self.normalization_stats_path.read_text())
        except OSError as exc:
            raise FileNotFoundError(
                f"Missing required RCSB train-only normalization stats: {self.normalization_stats_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid normalization stats JSON: {self.normalization_stats_path}") from exc
        if not isinstance(stats, dict):
            raise ValueError("RCSB normalization stats must be a JSON object")
        if stats.get("feature_count") != RCSB_PAIR_RAW_SCALAR_DIM:
            raise ValueError("RCSB normalization stats feature_count must be 89")
        if str(stats.get("fit_split", "")).lower() != "train":
            raise ValueError("RCSB normalization stats must declare fit_split='train'")
        expected_checksum = compute_train_uid_checksum(self.pair_manifest)
        checksum_fields = [
            stats[key]
            for key in ("train_uid_checksum", "train_unique_ids_sha256")
            if key in stats
        ]
        if not checksum_fields or any(value != expected_checksum for value in checksum_fields):
            raise ValueError("RCSB normalization stats train UID checksum does not match this manifest")
        mean = torch.as_tensor(stats.get("mean"), dtype=torch.float32)
        std = torch.as_tensor(stats.get("std"), dtype=torch.float32)
        if mean.shape != (RCSB_PAIR_RAW_SCALAR_DIM,) or std.shape != (RCSB_PAIR_RAW_SCALAR_DIM,):
            raise ValueError("RCSB normalization stats mean and std must each contain 89 values")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("RCSB normalization stats mean/std must be finite")
        if not bool((std > 0).all()):
            raise ValueError("RCSB normalization stats std values must be strictly positive")
        return mean, std

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_label_path(self, row: dict[str, Any]) -> Path:
        path = Path(str(row["label_path"]))
        return path if path.is_absolute() else self.pair_label_root / path

    def _load_label_record(self, row: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(row["pair_id"])
        path = self._resolve_label_path(row)
        offset = int(row["label_offset"])
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                line = handle.readline()
        except OSError as exc:
            raise FileNotFoundError(f"{pair_id}: cannot read JSONL labels at {path}:{offset}") from exc
        if not line or not line.endswith(b"\n"):
            raise ValueError(f"{pair_id}: label_offset does not point to a complete uncompressed JSONL line")
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{pair_id}: invalid JSONL record at {path}:{offset}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{pair_id}: JSONL record must be an object")
        expected = {
            "pair_id": row["pair_id"],
            "pdb_id": row["pdb_id"],
            "assembly_id": row["assembly_id"],
            "split": self.split,
            "left_seq_id": row["left_seq_id"],
            "right_seq_id": row["right_seq_id"],
            "contract": RCSB_LABEL_CONTRACT,
        }
        mismatches = {
            key: (str(record.get(key)), str(value))
            for key, value in expected.items()
            if str(record.get(key)) != str(value)
        }
        if mismatches:
            raise ValueError(f"{pair_id}: pair-label metadata mismatch: {mismatches}")
        return record

    def _feature_path(self, uid: str, *, raw_ps: bool) -> Path:
        prefix = uid[5:7]
        if raw_ps:
            return self.raw_ps_root / "pt" / prefix / f"{uid}.pt"
        return self.esm_feature_root / prefix / f"{uid}.pt"

    def _load_side(self, uid: str, expected_sha1: str, pair_id: str) -> dict[str, Any]:
        if self._feature_cache_pid != os.getpid():
            self._feature_cache.clear()
            self._feature_cache_pid = os.getpid()
        expected_sha1 = expected_sha1.lower()
        cached = self._feature_cache.get(uid)
        if cached is not None:
            cached_sha1, cached_item = cached
            if cached_sha1 != expected_sha1:
                raise ValueError(f"{pair_id}: manifest sequence SHA1 mismatch for cached UID {uid}")
            self._feature_cache.move_to_end(uid)
            return dict(cached_item)

        esm_path = self._feature_path(uid, raw_ps=False)
        raw_ps_path = self._feature_path(uid, raw_ps=True)
        if not esm_path.is_file():
            raise FileNotFoundError(f"{pair_id}: missing StarESMC feature for {uid}: {esm_path}")
        if not raw_ps_path.is_file():
            raise FileNotFoundError(f"{pair_id}: missing raw-PS feature for {uid}: {raw_ps_path}")
        esm = _torch_load(esm_path)
        raw_ps = _torch_load(raw_ps_path)
        if not isinstance(esm, dict) or not isinstance(raw_ps, dict):
            raise ValueError(f"{pair_id}: feature payloads for {uid} must be dictionaries")

        sequence = _sequence(esm.get("sequence"), f"{pair_id}/{uid} StarESMC")
        uid_match = _UID_RE.fullmatch(uid)
        assert uid_match is not None
        sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if sequence_sha256 != uid_match.group(1) or str(esm.get("sequence_sha256", "")).lower() != sequence_sha256:
            raise ValueError(f"{pair_id}: StarESMC sequence SHA256 does not match UID {uid}")
        if _sequence(raw_ps.get("sequence"), f"{pair_id}/{uid} raw-PS") != sequence:
            raise ValueError(f"{pair_id}: raw-PS sequence mismatch for {uid}")
        if str(raw_ps.get("sequence_sha256", "")).lower() != sequence_sha256:
            raise ValueError(f"{pair_id}: raw-PS sequence SHA256 mismatch for {uid}")
        sequence_sha1 = hashlib.sha1(sequence.encode("ascii")).hexdigest()
        if sequence_sha1 != expected_sha1:
            raise ValueError(f"{pair_id}: manifest sequence SHA1 mismatch for {uid}")

        if esm.get("repr_layers") != [12, 24, 36]:
            raise ValueError(f"{pair_id}: StarESMC repr_layers for {uid} must equal [12, 24, 36] in order")
        if esm.get("hidden_normalization") != "final_layernorm":
            raise ValueError(f"{pair_id}: StarESMC hidden_normalization for {uid} must be final_layernorm")
        if esm.get("resolved_revision") != RCSB_ESMC_RESOLVED_REVISION:
            raise ValueError(
                f"{pair_id}: StarESMC resolved_revision for {uid} must be {RCSB_ESMC_RESOLVED_REVISION}"
            )
        embeddings = torch.as_tensor(esm.get("embeddings"), dtype=torch.float32)
        if embeddings.shape != (len(sequence), 3456):
            raise ValueError(
                f"{pair_id}: StarESMC embeddings for {uid} have shape {tuple(embeddings.shape)}, "
                f"expected ({len(sequence)}, 3456)"
            )
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError(f"{pair_id}: StarESMC embeddings for {uid} contain non-finite values")

        if raw_ps.get("source") != "protenix_mini_esm_v0.5.0":
            raise ValueError(f"{pair_id}: raw-PS source for {uid} is not protenix_mini_esm_v0.5.0")
        physchem = torch.as_tensor(raw_ps.get("physchem_features"), dtype=torch.float32)
        spatial = torch.as_tensor(raw_ps.get("spatial_scalar_raw_features"), dtype=torch.float32)
        vector_payload = raw_ps.get("spatial_vector_features")
        if vector_payload is None:
            raise ValueError(f"{pair_id}: raw-PS spatial_vector_features missing for {uid}")
        vectors = torch.as_tensor(vector_payload, dtype=torch.float32)
        if physchem.shape != (len(sequence), 62):
            raise ValueError(f"{pair_id}: raw-PS physchem shape for {uid} must be [{len(sequence)}, 62]")
        if spatial.shape != (len(sequence), 89):
            raise ValueError(f"{pair_id}: raw-PS spatial scalar shape for {uid} must be [{len(sequence)}, 89]")
        if vectors.shape != (len(sequence), 8, 3):
            raise ValueError(f"{pair_id}: raw-PS spatial_vector_features shape for {uid} must be [{len(sequence)}, 8, 3]")
        for feature_name, tensor in (
            ("physchem_features", physchem),
            ("spatial_scalar_raw_features", spatial),
            ("spatial_vector_features", vectors),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{pair_id}: raw-PS {feature_name} for {uid} contain non-finite values")
        normalized_spatial = (spatial - self.scalar_mean) / self.scalar_std
        if not bool(torch.isfinite(normalized_spatial).all()):
            raise ValueError(f"{pair_id}: normalized raw-PS scalars for {uid} contain non-finite values")
        availability = torch.ones((len(sequence), 1), dtype=torch.float32)
        seq_features = torch.cat((normalized_spatial, availability), dim=-1)
        n_residues = len(sequence)
        positions = torch.arange(n_residues, dtype=torch.long)
        relative = positions.to(torch.float32) / max(1, n_residues - 1)
        item = {
            "pdb_id": uid,
            "source_pdb_id": "RCSB",
            "esm_embeddings": embeddings,
            "seq_features": seq_features,
            "chain_ids": torch.zeros(n_residues, dtype=torch.long),
            "chain_ids_raw": ["A"] * n_residues,
            "residue_names_1": list(sequence),
            "chain_pos": positions,
            "chain_rel_pos": relative,
            "protein_rel_pos": relative,
            "contact_edge_index": torch.empty((2, 0), dtype=torch.long),
            "contact_edge_scores": torch.empty((0,), dtype=torch.float32),
            "aux_contact_edge_index": torch.empty((2, 0), dtype=torch.long),
            "aux_contact_edge_scores": torch.empty((0,), dtype=torch.float32),
        }
        if self.feature_cache_size:
            self._feature_cache[uid] = (sequence_sha1, item)
            self._feature_cache.move_to_end(uid)
            while len(self._feature_cache) > self.feature_cache_size:
                self._feature_cache.popitem(last=False)
        return dict(item)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pair_id = str(row["pair_id"])
        record = self._load_label_record(row)
        sides = {}
        for side in ("left", "right"):
            uid = str(row[f"{side}_seq_id"])
            expected_sha1 = str(row[f"{side}_sequence_sha1"]).lower()
            sides[side] = self._load_side(uid, expected_sha1, pair_id)
            if len(sides[side]["residue_names_1"]) != int(row[f"{side}_length"]):
                raise ValueError(f"{pair_id}: {side} manifest length disagrees with feature sequence")

        raw_contacts = record.get("residue_contacts")
        if not isinstance(raw_contacts, list) or not raw_contacts:
            raise ValueError(f"{pair_id}: residue_contacts must be a nonempty Nx2 list")
        parsed_contacts: list[tuple[int, int]] = []
        seen_contacts: set[tuple[int, int]] = set()
        left_length = len(sides["left"]["residue_names_1"])
        right_length = len(sides["right"]["residue_names_1"])
        for contact in raw_contacts:
            if not isinstance(contact, list) or len(contact) != 2:
                raise ValueError(f"{pair_id}: residue_contacts must be a nonempty Nx2 list")
            left_index, right_index = contact
            if type(left_index) is not int or type(right_index) is not int:
                raise ValueError(f"{pair_id}: residue_contacts indices must have exact integer type")
            if not (0 <= left_index < left_length and 0 <= right_index < right_length):
                raise ValueError(f"{pair_id}: residue_contacts contains an out-of-range residue index")
            pair = (left_index, right_index)
            if pair in seen_contacts:
                raise ValueError(f"{pair_id}: residue_contacts contains duplicate residue pairs")
            seen_contacts.add(pair)
            parsed_contacts.append(pair)
        contacts = torch.tensor(parsed_contacts, dtype=torch.long)
        contact_index = contacts.T.contiguous()

        labels: dict[str, torch.Tensor] = {}
        observed_masks: dict[str, torch.Tensor] = {}
        positives: dict[str, list[int]] = {}
        for side in ("left", "right"):
            n_residues = len(sides[side]["residue_names_1"])
            observed_key = f"{side}_observed_indices"
            if record.get("observed_sequence_only") is True and int(row[f"{side}_observed"]) == n_residues:
                observed = torch.ones(n_residues, dtype=torch.bool)
            else:
                observed_indices = _indices(record.get(observed_key), observed_key, pair_id)
                observed = torch.zeros(n_residues, dtype=torch.bool)
                if observed_indices:
                    if min(observed_indices) < 0 or max(observed_indices) >= n_residues:
                        raise ValueError(f"{pair_id}: {observed_key} contains an out-of-range residue index")
                    observed[torch.tensor(observed_indices, dtype=torch.long)] = True
            positive_key = f"{side}_positive"
            positive_indices = _indices(record.get(positive_key), positive_key, pair_id)
            if positive_indices and (min(positive_indices) < 0 or max(positive_indices) >= n_residues):
                raise ValueError(f"{pair_id}: {positive_key} contains an out-of-range residue index")
            label = torch.full((n_residues,), -100, dtype=torch.long)
            label[observed] = 0
            if positive_indices:
                positive_tensor = torch.tensor(positive_indices, dtype=torch.long)
                if not bool(observed[positive_tensor].all()):
                    raise ValueError(f"{pair_id}: {side} positives include unobserved residues")
                label[positive_tensor] = 1
            observed_masks[side] = observed
            labels[side] = label
            positives[side] = positive_indices
            if int(observed.sum()) != int(row[f"{side}_observed"]):
                raise ValueError(f"{pair_id}: {side} observed-residue count disagrees with manifest")
            if len(positive_indices) != int(row[f"{side}_positive"]):
                raise ValueError(f"{pair_id}: {side} positive count disagrees with manifest")

        if contact_index.numel():
            if int(contact_index[0].min()) < 0 or int(contact_index[0].max()) >= labels["left"].numel():
                raise ValueError(f"{pair_id}: left contact index is out of range")
            if int(contact_index[1].min()) < 0 or int(contact_index[1].max()) >= labels["right"].numel():
                raise ValueError(f"{pair_id}: right contact index is out of range")
            if not bool(observed_masks["left"][contact_index[0]].all()) or not bool(
                observed_masks["right"][contact_index[1]].all()
            ):
                raise ValueError(f"{pair_id}: contact includes an unobserved residue")
            if set(torch.unique(contact_index[0]).tolist()) != set(positives["left"]):
                raise ValueError(f"{pair_id}: left site labels are not the projection of residue_contacts")
            if set(torch.unique(contact_index[1]).tolist()) != set(positives["right"]):
                raise ValueError(f"{pair_id}: right site labels are not the projection of residue_contacts")
        elif positives["left"] or positives["right"]:
            raise ValueError(f"{pair_id}: positive site labels exist without residue_contacts")
        if int(contact_index.shape[1]) != int(row["n_contacts"]):
            raise ValueError(f"{pair_id}: contact count disagrees with manifest")
        valid_cells = int(observed_masks["left"].sum()) * int(observed_masks["right"].sum())
        if valid_cells != int(row["n_valid_pair_cells"]):
            raise ValueError(f"{pair_id}: valid pair-cell count disagrees with manifest")

        sides["left"]["labels"] = labels["left"]
        sides["right"]["labels"] = labels["right"]
        return {
            "pair_id": pair_id,
            "pdb_id": str(row["pdb_id"]),
            "left": sides["left"],
            "right": sides["right"],
            "contact_index": contact_index,
            "left_observed_mask": observed_masks["left"],
            "right_observed_mask": observed_masks["right"],
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
