"""Fail closed before launch; full training additionally requires a valid smoke."""
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    dest, mode = Path(sys.argv[1]), sys.argv[2]
    receipt = json.loads((dest / "deployment.json").read_text())
    for name, expected in receipt["file_sha256"].items():
        if hashlib.sha256((dest / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Deployment changed: {name}")
    if mode != "full":
        return
    import torch
    import yaml

    smoke = dest / "runs/smoke"
    metrics = json.loads((smoke / "final_metrics.json").read_text())
    contract = json.loads((smoke / "config.json").read_text())
    expected_config = yaml.safe_load((dest / "configs/smoke.yaml").read_text())
    if contract["config"] != expected_config:
        raise ValueError("Smoke config mismatch")
    if contract["n_train_pairs"] != 256 or contract["n_val_pairs"] != 64:
        raise ValueError("Wrong smoke sample counts")
    if metrics["test"] is not None or metrics["n_test_pairs"] != "not_read":
        raise ValueError("Smoke violated sealed test contract")
    if not math.isfinite(metrics["best_val_chain_ap_macro"]):
        raise ValueError("Invalid smoke validation AP")
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(smoke / name, map_location="cpu", weights_only=False)
        if checkpoint["config"] != contract:
            raise ValueError("Checkpoint contract mismatch")
        if not checkpoint["model"] or not all(torch.isfinite(t).all().item() for t in checkpoint["model"].values()):
            raise ValueError("Nonfinite smoke model")
    approval = dict(passed=True, metrics=metrics, deployment_sha256=hashlib.sha256(
        (dest / "deployment.json").read_bytes()).hexdigest())
    (dest / "SMOKE_APPROVED.json").write_text(json.dumps(approval, indent=2) + "\n")
    print("Smoke passed; formal training starts from scratch, not smoke weights", flush=True)


if __name__ == "__main__":
    main()
