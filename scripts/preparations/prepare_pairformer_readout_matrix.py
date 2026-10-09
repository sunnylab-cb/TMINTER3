"""Isolate selective pooling, cross-chain outer initialization, and triangles."""
from copy import deepcopy
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[2]
base = yaml.safe_load((root / "configs/experiments/20261006_pairformer/PF1_triangle.yaml").read_text())
dest = root / "configs/experiments/20261009_pairformer_readout"
dest.mkdir(parents=True, exist_ok=True)
for name, rank, triangles in (("A_selective", 0, True), ("B_outer_triangle", 8, True), ("C_outer_no_triangle", 8, False)):
    config = deepcopy(base)
    config["pair_model"].update(pooling="softmax", outer_rank=rank, triangle_updates=triangles)
    config["training"]["output_dir"] = f"runs/pairformer_{name}_seed49_20261009_hpc2"
    config["metadata"].update(architecture=name, matrix_variant="A/B isolate outer initializer; B/C isolate triangles; same batching and losses")
    for smoke in (False, True):
        item = deepcopy(config)
        suffix = "_smoke" if smoke else ""
        if smoke:
            item["training"].update(epochs=1, num_workers=2)
            item["training"]["output_dir"] += suffix
            item["data"]["pair_manifest"] = "features/contact_labels_pairwise_5p5A_20261001/pair_manifest_smoke.csv"
        with (dest / f"{name}{suffix}.yaml").open("x") as handle:
            yaml.safe_dump(item, handle, sort_keys=False)
        print(name + suffix)
