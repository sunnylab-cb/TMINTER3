"""Materialize controlled full-RCSB model comparisons without changing the baseline."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


VARIANTS = [
    ("independent", {"mode": "independent", "n_heads": 8, "bottleneck": 64,
                     "dropout": 0.1, "max_partner_tokens": 32}, {},
     "Does partner conditioning improve over the shared monomer encoder?"),
    ("cross_attention", {"mode": "cross_attention", "n_heads": 8, "bottleneck": 64,
                         "dropout": 0.1, "max_partner_tokens": 32}, {},
     "Compare segment-pooled cross-attention against residue-pair outer products."),
    ("outer_null", {"pooling": "null_softmax"}, {},
     "Does a length-normalized null choice improve over forced partner pooling?"),
    ("outer_contact", {}, {"pair_contact_loss_weight": 0.1},
     "Does explicit 5.5-A pair-contact supervision improve the same outer-product model?"),
]


def build_matrix(baseline_path: Path, output_dir: Path) -> dict:
    raw = baseline_path.read_bytes()
    base = yaml.safe_load(raw)
    assert base["pair_model"]["mode"] == "outer_product"
    assert base["training"]["initialization"] == "scratch"
    assert base["training"]["evaluate_test"] is False
    output_dir.mkdir(parents=True, exist_ok=False)
    run_root = Path(base["training"]["output_dir"]).parent
    manifest = {"baseline_sha256": hashlib.sha256(raw).hexdigest(), "variants": []}
    for name, pair_updates, train_updates, question in VARIANTS:
        config = deepcopy(base)
        if name in {"independent", "cross_attention"}:
            config["pair_model"] = deepcopy(pair_updates)
        else:
            config["pair_model"].update(pair_updates)
        config["training"].update(train_updates)
        config["training"]["output_dir"] = str(run_root / f"rcsb_matrix_{name}_full_seed49_v1")
        config["metadata"].update(matrix="20261017_v1", variant=name, question=question)
        path = output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest["variants"].append({"name": name, "config": path.name,
                                     "output_dir": config["training"]["output_dir"],
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "question": question})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.baseline, args.output_dir), indent=2))
