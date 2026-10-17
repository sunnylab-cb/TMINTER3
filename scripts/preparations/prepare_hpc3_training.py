"""Rebase the frozen star baseline without changing its training contract."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import yaml


OLD_ROOT = "/data1/wholeset/RCSBprot"
MANIFEST_SHA = "4d908b57ecba298697852d0ed808c9ed48989dfd77439db42b0f97f2f7d31fa8"
ARCHIVE_SHA = "2f2cc672453bed0d4554b92ae21b8caa1f58ab31c631087bc068d959816ed606"


def rebase(source, root, deployment):
    full = copy.deepcopy(source)
    for key in ("pair_manifest", "pair_label_root", "rcsb_esm_feature_root",
                "rcsb_raw_ps_root", "rcsb_normalization_stats"):
        old = full["data"][key]
        if not old.startswith(OLD_ROOT + "/"):
            raise ValueError(f"Unexpected source path: {key}={old}")
        full["data"][key] = str(root) + old[len(OLD_ROOT):]
    full["training"]["output_dir"] = str(deployment / "runs/full")
    train = full["training"]
    if train["seed"] != 49 or train["epochs"] != 15:
        raise ValueError("Expected frozen seed49 / 15 epochs")
    if train["evaluate_test"] or train["initialization"] != "scratch":
        raise ValueError("Expected scratch training and sealed test")
    if any(k in train for k in ("max_train_pairs", "max_val_pairs", "smoke_max_chain_length")):
        raise ValueError("Formal source contains smoke caps")
    smoke = copy.deepcopy(full)
    smoke["training"].update(output_dir=str(deployment / "runs/smoke"), epochs=1,
                             max_train_pairs=256, max_val_pairs=64,
                             smoke_max_chain_length=512)
    return full, smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    root, dest = args.root.resolve(), args.deployment.resolve()
    audit = json.loads((root / "_esmc/completion_coverage_audit.json").read_text())
    if audit["expected"] != 214097 or audit["completed"] != 214097 or any(
        audit[k] for k in ("missing", "unexpected", "misplaced")
    ):
        raise ValueError("Incomplete ESMC coverage")
    manifest = root / "training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError("Ready manifest differs from frozen source")
    archive = root / "training/ppis_ge50_v1/launch_records/full_seed49_v1/code_at_submit.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Frozen code archive digest mismatch")
    dest.mkdir(parents=True, exist_ok=False)
    code = dest / "code"
    code.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(code, filter="data")
    source = yaml.safe_load((code / "configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml").read_text())
    full, smoke = rebase(source, root, dest)
    for key in ("pair_manifest", "pair_label_root", "rcsb_esm_feature_root",
                "rcsb_raw_ps_root", "rcsb_normalization_stats"):
        if not Path(full["data"][key]).exists():
            raise FileNotFoundError(full["data"][key])
    for directory in ("configs", "logs", "runs"):
        (dest / directory).mkdir()
    for name, config in (("full", full), ("smoke", smoke)):
        (dest / f"configs/{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    checker = dest / "check_hpc3_training.py"
    shutil.copyfile(Path(__file__).with_name(checker.name), checker)
    hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
              for base in (code, dest / "configs") for p in sorted(base.rglob("*")) if p.is_file()}
    hashes[checker.name] = hashlib.sha256(checker.read_bytes()).hexdigest()
    receipt = dict(archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   ready_manifest_sha256=MANIFEST_SHA, file_sha256=hashes,
                   test_policy="sealed; no test evaluation", source="star frozen full_seed49_v1")
    (dest / "deployment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
