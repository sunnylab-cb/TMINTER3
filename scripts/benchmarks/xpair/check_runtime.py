"""Check official X-PAIR with existing torch; synthetic tensors are not benchmark data."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


ASSETS = Path(os.environ.get("XPAIR_ASSETS", Path(__file__).resolve().parent)).resolve()
SOURCE = Path(os.environ.get("XPAIR_SOURCE", ASSETS.parent.parent / "official" / "X-PAIR-0.1.1")).resolve()
BASE_TORCH = Path(os.environ.get(
    "XPAIR_BASE_TORCH", "/home/husrcf/anaconda3/envs/AIAA/lib/python3.12/site-packages/torch"
)).resolve()
sys.path[:0] = [str(SOURCE), str(ASSETS / "runtime_vendor")]


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def torch_identity(torch):
    path = Path(torch.__file__).resolve()
    if path.parent != BASE_TORCH:
        raise RuntimeError(f"Unexpected torch import: {path}")
    return {
        "version": torch.__version__,
        "file": str(path),
        "init_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_checkpoint(path, torch, np):
    # Only allow the known NumPy scalar metadata in these official checkpoints.
    safe = [np._core.multiarray.scalar, np.dtype,
            type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def check_one(path, device, torch, np, model_class):
    ckpt = load_checkpoint(path, torch, np)
    hparams = ckpt["hyper_parameters"]
    model = model_class(hparams).eval().to(device)
    result = model.load_state_dict(ckpt["state_dict"], strict=True)
    torch.manual_seed(123)
    x1 = torch.randn(2, 64, model.emb_dim, device=device)
    x2 = torch.randn(2, 93, model.emb_dim, device=device)
    m1 = torch.arange(64, device=device)[None, :] < torch.tensor([64, 49], device=device)[:, None]
    m2 = torch.arange(93, device=device)[None, :] < torch.tensor([93, 73], device=device)[:, None]
    batch = {"input1": (x1, m1), "input2": (x2, m2)}
    task = hparams["model_task"]
    with torch.inference_mode():
        out = model(batch, task=task)
        swapped = model({"input1": (x2, m2), "input2": (x1, m1)}, task=task)
        require(out[1].shape == m1.shape and out[2].shape == m2.shape,
                "Unexpected residue output shapes")
        require(torch.isfinite(out[1][m1]).all() and torch.isfinite(out[2][m2]).all(),
                "Non-finite residue logits")
        swap_error = max((out[1][m1] - swapped[2][m1]).abs().max().item(),
                         (out[2][m2] - swapped[1][m2]).abs().max().item())
        if task == "multitask":
            require(out[0].shape == (2,) and torch.isfinite(out[0]).all(),
                    "Invalid interaction logits")
            swap_error = max(swap_error, (out[0] - swapped[0]).abs().max().item())
        single = model({"input1": (x1[1:2, :49], m1[1:2, :49]),
                        "input2": (x2[1:2, :73], m2[1:2, :73])}, task=task)
        padding_error = max((out[1][1, :49] - single[1][0]).abs().max().item(),
                            (out[2][1, :73] - single[2][0]).abs().max().item())
        if task == "multitask":
            padding_error = max(padding_error, (out[0][1] - single[0][0]).abs().item())
        require(swap_error < 1e-4, f"Swap error: {swap_error}")
        require(padding_error < 1e-4, f"Padding error: {padding_error}")
    return {
        "checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task": task, "plm": hparams["plm"], "strict_load": str(result),
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic_shapes": [list(out[1].shape), list(out[2].shape)],
        "swap_max_abs_error": swap_error, "padding_max_abs_error": padding_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    identity = torch_identity(torch)
    import xpair.model as official_model
    require(Path(official_model.__file__).resolve() == SOURCE / "xpair/model.py",
            f"Unexpected X-PAIR source: {official_model.__file__}")
    XPairModel = official_model.XPairModel
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    checkpoints = [ASSETS / "interface_pioneer.ckpt",
                   SOURCE / "xpair/default_model/multitask_xfair.ckpt"]
    checks = [check_one(p, device, torch, np, XPairModel) for p in checkpoints]
    require(torch_identity(torch) == identity, "Torch identity changed")
    report = {
        "scope": "synthetic compatibility checks only; no biological metrics",
        "torch_unchanged": identity, "device": str(device),
        "source": str(SOURCE), "runtime_vendor": str(ASSETS / "runtime_vendor"),
        "support_packages": {n: importlib.metadata.version(n) for n in
                             ("pytorch-lightning", "torchmetrics", "lightning-utilities")},
        "checks": checks,
    }
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        report["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
