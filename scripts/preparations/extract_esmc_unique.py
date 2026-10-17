#!/usr/bin/env python3
"""Extract residue-aligned ESM-C MLC embeddings for unique sequences.

The model is loaded once per process and sequences are length-bucketed into
padded batches.  Outputs are addressed by exact sequence SHA-256 and written
atomically, so interrupted runs can safely resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch


def _sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: no rows")
    required = {"unique_id", "seq_sha256", "n_residues", "sequence"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        sequence = row["sequence"].strip().upper()
        unique_id = row["unique_id"].strip()
        digest = row["seq_sha256"].strip().lower()
        n_residues = int(row["n_residues"])
        if not sequence or len(sequence) != n_residues:
            raise ValueError(f"{unique_id}: sequence length {len(sequence)} != {n_residues}")
        if _sha256(sequence) != digest:
            raise ValueError(f"{unique_id}: sequence SHA-256 mismatch")
        if unique_id in seen_ids or digest in seen_hashes:
            raise ValueError(f"duplicate unique sequence row: {unique_id} / {digest}")
        seen_ids.add(unique_id)
        seen_hashes.add(digest)
        normalized.append({**row, "unique_id": unique_id, "seq_sha256": digest, "n_residues": n_residues, "sequence": sequence})
    return sorted(normalized, key=lambda row: (row["n_residues"], row["seq_sha256"]))


def _output_path(root: Path, row: dict[str, Any]) -> Path:
    return root / row["seq_sha256"][:2] / f"{row['unique_id']}.pt"


def _load_existing(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _valid_existing(
    path: Path,
    row: dict[str, Any],
    layers: list[int],
    embedding_dim: int,
    hidden_normalization: str,
    revision: str,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = _load_existing(path)
        embedding = torch.as_tensor(payload["embeddings"])
        return (
            payload.get("sequence_sha256") == row["seq_sha256"]
            and payload.get("sequence") == row["sequence"]
            and list(payload.get("repr_layers", [])) == layers
            and payload.get("hidden_normalization") == hidden_normalization
            and payload.get("resolved_revision") == revision
            and tuple(embedding.shape) == (row["n_residues"], embedding_dim)
            and not embedding.requires_grad
            and bool(torch.isfinite(embedding).all())
        )
    except Exception:
        return False


def _make_batches(rows: list[dict[str, Any]], max_tokens: int, max_batch_size: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max = 0
    for row in rows:
        padded_length = int(row["n_residues"]) + 2
        candidate_max = max(current_max, padded_length)
        candidate_size = len(current) + 1
        if current and (candidate_max * candidate_size > max_tokens or candidate_size > max_batch_size):
            batches.append(current)
            current = []
            current_max = 0
        current.append(row)
        current_max = max(current_max, padded_length)
    if current:
        batches.append(current)
    return batches


def _load_model(args: argparse.Namespace):
    os.environ["HF_HOME"] = str(args.cache_dir)
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from huggingface_hub import load_torch_model, snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    weight_files = sorted((snapshot_path / "data" / "weights").glob("*.pth"))
    if len(weight_files) != 1:
        raise RuntimeError(f"expected one ESM-C weight file under {snapshot_path}, found {weight_files}")
    model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=args.use_flash_attn,
    ).eval()
    incompatible = load_torch_model(
        model,
        weight_files[0],
        strict=True,
        weights_only=True,
        map_location="cpu",
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible checkpoint: {incompatible}")
    dtype = getattr(torch, args.dtype)
    model = model.to(device=torch.device(args.device), dtype=dtype).eval()
    return model, snapshot_path, weight_files[0]


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="biohub/esmc-600m-2024-12")
    parser.add_argument("--revision", default="e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--layers", default="12,24,36")
    parser.add_argument(
        "--hidden-normalization",
        choices=["final_layernorm", "none"],
        default="final_layernorm",
        help="Normalize each selected residual stream with the pretrained final LayerNorm.",
    )
    parser.add_argument("--max-tokens-per-batch", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tag", default="worker0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("require 0 <= shard-index < shard-count")
    if args.max_tokens_per_batch <= 0 or args.max_batch_size <= 0:
        raise ValueError("batch limits must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but torch.cuda.is_available() is false")
    layers = sorted({int(value.strip()) for value in args.layers.split(",") if value.strip()})
    if not layers or min(layers) < 1 or max(layers) > 36:
        raise ValueError(f"invalid 1-based layers: {layers}")
    embedding_dim = 1152 * len(layers)

    all_rows = _read_manifest(args.manifest)
    eligible_rows = [
        row
        for row in all_rows
        if row["n_residues"] >= args.min_length
        and (args.max_length is None or row["n_residues"] <= args.max_length)
    ]
    rows = [row for index, row in enumerate(eligible_rows) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    skipped = 0
    pending: list[dict[str, Any]] = []
    for row in rows:
        path = _output_path(args.output_root, row)
        if not args.overwrite and _valid_existing(
            path, row, layers, embedding_dim, args.hidden_normalization, args.revision
        ):
            skipped += 1
        else:
            pending.append(row)
    batches = _make_batches(pending, args.max_tokens_per_batch, args.max_batch_size)
    print(
        json.dumps(
            {
                "tag": args.tag,
                "manifest_rows": len(all_rows),
                "eligible_rows": len(eligible_rows),
                "shard_rows": len(rows),
                "valid_existing": skipped,
                "pending": len(pending),
                "batches": len(batches),
                "max_tokens_per_batch": args.max_tokens_per_batch,
            }
        ),
        flush=True,
    )
    if not pending:
        return 0

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    load_started = time.time()
    model, snapshot_path, weight_file = _load_model(args)
    load_seconds = time.time() - load_started
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    selected_indices = torch.tensor([layer - 1 for layer in layers], device=args.device)
    output_dtype = getattr(torch, args.output_dtype)
    completed = 0
    completed_residues = 0
    inference_seconds = 0.0
    wall_started = time.time()

    for batch_index, batch in enumerate(batches, 1):
        sequences = [row["sequence"] for row in batch]
        tokens = model._tokenize(sequences)
        if any(int((tokens[i] != model.tokenizer.pad_token_id).sum()) != len(sequence) + 2 for i, sequence in enumerate(sequences)):
            raise RuntimeError("ESM-C tokenization length mismatch")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        started = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=getattr(torch, args.dtype)):
            output = model(sequence_tokens=tokens)
            if output.hidden_states is None or tuple(output.hidden_states.shape[:2]) != (36, len(batch)):
                raise RuntimeError(f"unexpected hidden-state shape {getattr(output.hidden_states, 'shape', None)}")
            selected = output.hidden_states.index_select(0, selected_indices)
            if args.hidden_normalization == "final_layernorm":
                selected = model.transformer.norm(selected)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        batch_seconds = time.time() - started
        inference_seconds += batch_seconds
        for sequence_index, row in enumerate(batch):
            length = int(row["n_residues"])
            embedding = (
                selected[:, sequence_index, 1 : length + 1]
                .permute(1, 0, 2)
                .reshape(length, embedding_dim)
                .detach()
                .to(dtype=output_dtype)
                .cpu()
                .contiguous()
            )
            if tuple(embedding.shape) != (length, embedding_dim) or not bool(torch.isfinite(embedding).all()):
                raise RuntimeError(f"{row['unique_id']}: invalid embedding {tuple(embedding.shape)}")
            payload = {
                "embeddings": embedding,
                "sequence": row["sequence"],
                "sequence_sha256": row["seq_sha256"],
                "unique_id": row["unique_id"],
                "n_residues": length,
                "model_name": "esmc_600m_2024_12",
                "repo_id": args.repo_id,
                "requested_revision": args.revision,
                "resolved_revision": snapshot_path.name,
                "weight_file_name": weight_file.name,
                "repr_layers": layers,
                "repr_layers_indexing": "1_based_transformer_blocks",
                "embedding_combine": "concat",
                "hidden_normalization": args.hidden_normalization,
                "embedding_dim": embedding_dim,
                "model_dtype": args.dtype,
                "embedding_dtype": str(embedding.dtype),
                "flash_attention_active": bool(model._use_flash_attn),
            }
            _atomic_save(payload, _output_path(args.output_root, row))
            completed += 1
            completed_residues += length
        del output, selected, tokens
        elapsed = time.time() - wall_started
        rate = completed_residues / max(inference_seconds, 1e-9)
        peak_allocated = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0
        print(
            f"[{args.tag} batch {batch_index}/{len(batches)}] sequences={completed}/{len(pending)} "
            f"residues={completed_residues} batch_size={len(batch)} max_len={max(len(x) for x in sequences)} "
            f"batch_s={batch_seconds:.2f} infer_res_s={rate:.1f} wall_s={elapsed:.1f} "
            f"peak_alloc_gib={peak_allocated / 2**30:.2f}",
            flush=True,
        )

    summary = {
        "tag": args.tag,
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "shard_rows": len(rows),
        "valid_existing_at_start": skipped,
        "completed": completed,
        "completed_residues": completed_residues,
        "layers": layers,
        "hidden_normalization": args.hidden_normalization,
        "embedding_dim": embedding_dim,
        "model_dtype": args.dtype,
        "output_dtype": args.output_dtype,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.time() - wall_started,
        "esm_version": importlib.metadata.version("esm"),
        "torch_version": torch.__version__,
        "flash_attention_active": bool(model._use_flash_attn),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
