"""Checkpoint primitives for exact continuation from trusted local checkpoints."""

import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Replace path only after writing a complete sibling temporary file.

    The parent directory must exist. Payload construction (including RNG capture)
    belongs to the caller; saving does not modify it or consume training RNGs.
    """
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy, CPU torch, and every available CUDA device RNG."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a captured snapshot; reject incompatible CUDA device counts."""
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state requires the same CUDA device count")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda_state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def require_matching_config(saved: dict, current: dict) -> None:
    """Require exact dictionary equality, including nested metadata and all keys."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise ValueError("Checkpoint and current configuration must both be dictionaries")
    try:
        matches = bool(saved == current)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint configuration cannot be compared") from error
    if not matches:
        raise ValueError("Checkpoint configuration does not match current configuration")
