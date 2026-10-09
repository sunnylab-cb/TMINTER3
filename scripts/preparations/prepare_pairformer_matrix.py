"""Create paired architecture trials; retain all training/validation chains."""
from copy import deepcopy
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[2]
base = yaml.safe_load((root / "configs/experiments/20261002_pairwise_followup/S5_site_only.yaml").read_text())
dest = root / "configs/experiments/20261006_pairformer"
dest.mkdir(parents=True, exist_ok=True)
for name, triangles in (("PF0_transition", False), ("PF1_triangle", True)):
    config = deepcopy(base)
    config["pair_model"] = dict(mode="pairformer", pair_hidden=32, triangle_hidden=16,
                                pair_chunk=128, pairformer_blocks=2, triangle_updates=triangles,
                                checkpoint_blocks=True)
    config["training"].update(max_pair_cells=4000000, max_single_pair_cells=25000000)
    config["training"]["output_dir"] = f"runs/pairformer_{name}_seed49_20261006_hpc2"
    config["metadata"].update(architecture=name, matrix_variant="PF0/PF1 differ only by triangle multiplication",
                              pair_input="joint AA/AB/BA/BB grid; intrachain ESM2 contacts; no experimental interchain features")
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
