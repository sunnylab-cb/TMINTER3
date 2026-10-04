#!/usr/bin/env python3
"""Build partner-specific residue labels for chain pairs in a frozen split.

Only chain pairs with at least one 5.5 A heavy-atom contact are materialized by
default.  This matches the intended use case: residue-site prediction for a
known protein pair, rather than protein-pair interaction classification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _heavy_atom(name: Any) -> bool:
    atom = _text(name).upper()
    return bool(atom) and not atom.startswith(("H", "D"))


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_chain_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    by_pdb: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = _text(row.get("pdb_id")).lower()
            chain_id = _text(row.get("chain_id"))
            seq_id = _text(row.get("seq_id"))
            split = _text(row.get("split")).lower()
            if not pdb_id or not chain_id or not seq_id or split not in {"train", "val", "test"}:
                continue
            by_pdb[pdb_id].append(row)
    if not by_pdb:
        raise ValueError(f"No usable rows in {path}")
    return by_pdb


def _chain_residue_rows(chain_ids: list[str], chain_id: str) -> np.ndarray:
    return np.asarray([idx for idx, value in enumerate(chain_ids) if value == chain_id], dtype=np.int64)


def _chain_atoms(
    all_atom_coords: list[Any],
    all_atom_names: list[Any],
    residue_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_out: list[np.ndarray] = []
    local_rows_out: list[np.ndarray] = []
    observed = np.zeros(residue_rows.size, dtype=bool)
    for local_row, global_row in enumerate(residue_rows.tolist()):
        coords = np.asarray(all_atom_coords[global_row], dtype=np.float32)
        names = list(all_atom_names[global_row])
        if coords.ndim != 2 or coords.shape[0] == 0:
            continue
        keep = np.asarray([_heavy_atom(name) for name in names], dtype=bool)
        coords = coords[keep]
        coords = coords[np.isfinite(coords).all(axis=1)]
        if coords.shape[0] == 0:
            continue
        observed[local_row] = True
        coords_out.append(coords)
        local_rows_out.append(np.full(coords.shape[0], local_row, dtype=np.int64))
    if not coords_out:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            observed,
        )
    return np.concatenate(coords_out), np.concatenate(local_rows_out), observed


def _aabb_min_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.maximum(
        0.0,
        np.maximum(left.min(axis=0) - right.max(axis=0), right.min(axis=0) - left.max(axis=0)),
    )
    return float(np.sqrt(np.sum(delta * delta)))


def _pair_labels(
    left_coords: np.ndarray,
    left_atom_rows: np.ndarray,
    left_observed: np.ndarray,
    left_len: int,
    right_coords: np.ndarray,
    right_atom_rows: np.ndarray,
    right_observed: np.ndarray,
    right_len: int,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_observed_tensor = torch.as_tensor(left_observed, dtype=torch.bool)
    right_observed_tensor = torch.as_tensor(right_observed, dtype=torch.bool)
    left_labels = torch.full((left_len,), -100, dtype=torch.long)
    right_labels = torch.full((right_len,), -100, dtype=torch.long)
    left_labels[left_observed_tensor] = 0
    right_labels[right_observed_tensor] = 0
    empty_contacts = torch.empty((2, 0), dtype=torch.long)
    if not left_coords.size or not right_coords.size or _aabb_min_distance(left_coords, right_coords) > cutoff:
        return left_labels, right_labels, empty_contacts
    neighbors = cKDTree(left_coords).query_ball_tree(cKDTree(right_coords), r=cutoff)
    residue_contacts: set[tuple[int, int]] = set()
    for left_atom, right_atoms in enumerate(neighbors):
        if not right_atoms:
            continue
        left_residue = int(left_atom_rows[left_atom])
        for right_residue in np.unique(right_atom_rows[np.asarray(right_atoms)]):
            residue_contacts.add((left_residue, int(right_residue)))
    if not residue_contacts:
        return left_labels, right_labels, empty_contacts
    ordered_contacts = sorted(residue_contacts)
    contact_index = torch.tensor(ordered_contacts, dtype=torch.long).T.contiguous()
    left_positive = torch.unique(contact_index[0])
    right_positive = torch.unique(contact_index[1])
    if left_positive.numel():
        left_labels[left_positive] = 1
    if right_positive.numel():
        right_labels[right_positive] = 1
    return left_labels, right_labels, contact_index


def _process_pdb(
    pdb_id: str,
    rows: list[dict[str, str]],
    output_root: str,
    cutoff: float,
    include_noninteracting: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"pdb_id": pdb_id, "status": "ERROR", "pairs": [], "error": ""}
    try:
        split_values = {_text(row.get("split")).lower() for row in rows}
        if len(split_values) != 1:
            raise ValueError(f"complex crosses splits: {sorted(split_values)}")
        split = next(iter(split_values))
        feature_paths = {_text(row.get("feature_path")) for row in rows if _text(row.get("feature_path"))}
        if len(feature_paths) != 1:
            raise ValueError(f"expected one feature_path, got {sorted(feature_paths)}")
        feature_path = Path(next(iter(feature_paths)))
        feature = _torch_load(feature_path)
        chain_ids = [_text(value) for value in _as_list(feature["chain_ids"])]
        all_atom_coords = list(feature["all_atom_coords"])
        all_atom_names = list(feature["all_atom_names"])
        residue_names = [_text(value).upper() for value in _as_list(feature["residue_names"])]

        prepared: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda value: (_text(value.get("chain_id")), _text(value.get("seq_id")))):
            chain_id = _text(row["chain_id"])
            residue_rows = _chain_residue_rows(chain_ids, chain_id)
            expected = int(row.get("n_residues") or row.get("len_seq") or 0)
            if expected and residue_rows.size != expected:
                raise ValueError(
                    f"{row['seq_id']}: feature chain length {residue_rows.size} != manifest length {expected}"
                )
            coords, atom_rows, observed = _chain_atoms(all_atom_coords, all_atom_names, residue_rows)
            sequence = "".join(AA3_TO_1.get(residue_names[idx], "X") for idx in residue_rows.tolist())
            sequence_sha1 = hashlib.sha1(sequence.encode("ascii")).hexdigest()
            expected_sha1 = _text(row.get("seq_sha1")).lower()
            if expected_sha1 and sequence_sha1 != expected_sha1:
                raise ValueError(
                    f"{row['seq_id']}: structure sequence SHA1 {sequence_sha1} != manifest {expected_sha1}"
                )
            prepared[_text(row["seq_id"])] = {
                "row": row,
                "chain_id": chain_id,
                "length": int(residue_rows.size),
                "coords": coords,
                "atom_rows": atom_rows,
                "observed": observed,
                "sequence": sequence,
                "sequence_sha1": sequence_sha1,
            }

        output = Path(output_root)
        pair_rows: list[dict[str, Any]] = []
        n_zero = 0
        for left_id, right_id in combinations(sorted(prepared), 2):
            left = prepared[left_id]
            right = prepared[right_id]
            left_labels, right_labels, contact_index = _pair_labels(
                left["coords"],
                left["atom_rows"],
                left["observed"],
                left["length"],
                right["coords"],
                right["atom_rows"],
                right["observed"],
                right["length"],
                cutoff,
            )
            n_left = int((left_labels == 1).sum().item())
            n_right = int((right_labels == 1).sum().item())
            if n_left == 0 or n_right == 0:
                n_zero += 1
                if not include_noninteracting:
                    continue
            pair_id = f"{left_id}--{right_id}"
            relative_path = Path(split) / pdb_id / f"{_safe_id(pair_id)}.pt"
            pair_path = output / relative_path
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            # Contact computation has already been performed, so always replace
            # the artifact atomically.  This prevents an old cutoff/source file
            # from silently disagreeing with the freshly generated manifest.
            payload = {
                "pair_id": pair_id,
                "pdb_id": pdb_id,
                "split": split,
                "left_seq_id": left_id,
                "right_seq_id": right_id,
                "left_chain_id": left["chain_id"],
                "right_chain_id": right["chain_id"],
                "left_labels": left_labels,
                "right_labels": right_labels,
                "left_observed_mask": torch.as_tensor(left["observed"], dtype=torch.bool),
                "right_observed_mask": torch.as_tensor(right["observed"], dtype=torch.bool),
                "contact_index": contact_index,
                "left_sequence_sha1": left["sequence_sha1"],
                "right_sequence_sha1": right["sequence_sha1"],
                "cutoff_angstrom": float(cutoff),
                "heavy_only": True,
                "label_definition": "partner-specific heavy-atom contact within cutoff",
                "source_feature_path": str(feature_path),
            }
            temporary = pair_path.parent / f".{pair_path.name}.{uuid.uuid4().hex}.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, pair_path)
            pair_rows.append(
                {
                    "split": split,
                    "pair_id": pair_id,
                    "pdb_id": pdb_id,
                    "left_seq_id": left_id,
                    "right_seq_id": right_id,
                    "left_chain_id": left["chain_id"],
                    "right_chain_id": right["chain_id"],
                    "left_length": left["length"],
                    "right_length": right["length"],
                    "left_positive": n_left,
                    "right_positive": n_right,
                    "left_observed": int(np.sum(left["observed"])),
                    "right_observed": int(np.sum(right["observed"])),
                    "left_sequence_sha1": left["sequence_sha1"],
                    "right_sequence_sha1": right["sequence_sha1"],
                    "n_contacts": int(contact_index.shape[1]),
                    "n_valid_pair_cells": int(np.sum(left["observed"]) * np.sum(right["observed"])),
                    "label_path": str(relative_path),
                }
            )
        result.update(status="OK", pairs=pair_rows, n_candidate_pairs=len(prepared) * (len(prepared) - 1) // 2, n_zero=n_zero)
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cutoff", type=float, default=5.5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--include-noninteracting-pairs", action="store_true")
    parser.add_argument("--max-pdbs", type=int, default=None)
    parser.add_argument("--smoke-train-pairs", type=int, default=0)
    parser.add_argument("--smoke-eval-pairs", type=int, default=0)
    args = parser.parse_args()

    by_pdb = _load_chain_manifest(args.chain_manifest)
    pdb_items = sorted(by_pdb.items())
    if args.max_pdbs is not None:
        pdb_items = pdb_items[: args.max_pdbs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (pdb_id, rows, str(args.output_root), args.cutoff, args.include_noninteracting_pairs)
        for pdb_id, rows in pdb_items
    ]
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        results = [_process_pdb(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_pdb, *task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda row: row["pdb_id"])
    pair_rows = [pair for result in results for pair in result.get("pairs", [])]
    pair_rows.sort(key=lambda row: (row["split"], row["pdb_id"], row["pair_id"]))
    manifest_path = args.output_root / "pair_manifest.csv"
    fields = [
        "split", "pair_id", "pdb_id", "left_seq_id", "right_seq_id", "left_chain_id", "right_chain_id",
        "left_length", "right_length", "left_positive", "right_positive",
        "left_observed", "right_observed", "left_sequence_sha1", "right_sequence_sha1",
        "n_contacts", "n_valid_pair_cells", "label_path",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    smoke_manifest_path: Path | None = None
    if args.smoke_train_pairs > 0 or args.smoke_eval_pairs > 0:
        limits = {
            "train": max(0, int(args.smoke_train_pairs)),
            "val": max(0, int(args.smoke_eval_pairs)),
            "test": max(0, int(args.smoke_eval_pairs)),
        }
        seen: Counter[str] = Counter()
        smoke_rows: list[dict[str, Any]] = []
        for row in pair_rows:
            split = str(row["split"])
            if seen[split] < limits.get(split, 0):
                smoke_rows.append(row)
                seen[split] += 1
        smoke_manifest_path = args.output_root / "pair_manifest_smoke.csv"
        with smoke_manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(smoke_rows)
    errors = [result for result in results if result["status"] != "OK"]
    split_counts = Counter(row["split"] for row in pair_rows)
    summary = {
        "contract": "known interacting pair; partner-specific 5.5 A heavy-atom labels; both chains in frozen universe",
        "chain_manifest": str(args.chain_manifest),
        "output_root": str(args.output_root),
        "pair_manifest": str(manifest_path),
        "smoke_pair_manifest": str(smoke_manifest_path) if smoke_manifest_path else None,
        "cutoff_angstrom": args.cutoff,
        "heavy_only": True,
        "include_noninteracting_pairs": args.include_noninteracting_pairs,
        "n_pdbs": len(results),
        "n_pdb_errors": len(errors),
        "n_candidate_pairs": sum(int(result.get("n_candidate_pairs", 0)) for result in results),
        "n_zero_contact_pairs": sum(int(result.get("n_zero", 0)) for result in results),
        "n_materialized_pairs": len(pair_rows),
        "pairs_by_split": dict(split_counts),
        "directional_residues": sum(int(row["left_length"]) + int(row["right_length"]) for row in pair_rows),
        "directional_positives": sum(int(row["left_positive"]) + int(row["right_positive"]) for row in pair_rows),
        "directional_observed_residues": sum(
            int(row["left_observed"]) + int(row["right_observed"]) for row in pair_rows
        ),
        "residue_pair_contacts": sum(int(row["n_contacts"]) for row in pair_rows),
        "valid_residue_pair_cells": sum(int(row["n_valid_pair_cells"]) for row in pair_rows),
        "errors": [{"pdb_id": row["pdb_id"], "error": row["error"]} for row in errors[:50]],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
