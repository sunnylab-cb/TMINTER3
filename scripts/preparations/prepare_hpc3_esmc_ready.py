"""Build exact ready-pair ESMC inputs; no re-split, re-label or sequence trimming."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


FIELDS = ["unique_id", "seq_sha256", "n_residues", "sequence"]
EXPECTED_PAIRS = {"train": 1150993, "val": 123863, "test": 131829}
EXPECTED_UNIQUE = {"train": 170465, "val": 21662, "test": 21970}


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root, out, shards=4):
    if shards < 1:
        raise ValueError("shards must be positive")
    manifest = root / "training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv"
    registry = root / "sequences/observed_full_success_v1/unique_sequences.tsv"
    receipt = json.loads((root / "_transfer/metadata_snapshot.json").read_text())
    if sha_file(manifest) != receipt["ready_sha256"]:
        raise ValueError("Ready manifest differs from checked transfer snapshot")
    counts, uid_split, lengths = Counter(), {}, {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            counts[split] += 1
            for side in ("left", "right"):
                uid = row[side + "_seq_id"]
                length = int(row[side + "_length"])
                if uid_split.setdefault(uid, split) != split:
                    raise ValueError("Cross-split exact sequence " + uid)
                if lengths.setdefault(uid, length) != length:
                    raise ValueError("Inconsistent manifest length " + uid)
    unique_counts = dict(Counter(uid_split.values()))
    if dict(counts) != EXPECTED_PAIRS or unique_counts != EXPECTED_UNIQUE:
        raise ValueError("Input differs from frozen ready-v1 counts")
    selected = {}
    with registry.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            uid = row["unique_id"]
            if uid not in uid_split:
                continue
            sequence = row["sequence"]
            digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            if uid in selected or uid != "useq_" + digest or row["seq_sha256"] != digest:
                raise ValueError("Duplicate/mismatched sequence ID " + uid)
            if len(sequence) != lengths[uid] or int(row["n_residues"]) != len(sequence):
                raise ValueError("Registry length mismatch " + uid)
            if sequence != sequence.strip().upper():
                raise ValueError("Noncanonical sequence string " + uid)
            selected[uid] = {k: row[k] for k in FIELDS}
    missing = set(uid_split) - set(selected)
    if missing:
        raise ValueError("Missing {} ready sequences".format(len(missing)))
    rows = sorted(selected.values(), key=lambda r: (int(r["n_residues"]), r["seq_sha256"]))
    out.mkdir(parents=True, exist_ok=False)
    def write(name, items):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(items)
    write("all_ready_unique.tsv", rows)
    shard_stats = []
    for index in range(shards):
        part = rows[index::shards]
        name = "shard_{:02d}.tsv".format(index)
        write(name, part)
        shard_stats.append({"file": name, "sequences": len(part),
                            "residues": sum(int(r["n_residues"]) for r in part),
                            "sum_length_squared": sum(int(r["n_residues"]) ** 2 for r in part),
                            "sha256": sha_file(out / name)})
    smoke = {}
    for target in [100, 256, 512, 1024, 2048, int(rows[-1]["n_residues"])]:
        row = min(rows, key=lambda r: abs(int(r["n_residues"]) - target))
        smoke[row["unique_id"]] = row
    write("smoke.tsv", sorted(smoke.values(), key=lambda r: int(r["n_residues"])))
    result = {"ready_pairs_by_split": dict(counts), "unique_by_split": unique_counts,
              "total_unique": len(rows), "total_residues": sum(lengths.values()),
              "max_length": max(lengths.values()), "min_length": min(lengths.values()),
              "pair_manifest_sha256": sha_file(manifest), "registry_sha256": sha_file(registry),
              "model_revision": "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c",
              "layers": [12, 24, 36], "normalization": "final_layernorm",
              "embedding_dim": 3456, "output_dtype": "float32",
              "shards": shard_stats, "smoke_sequences": len(smoke),
              "scope": "all exact sequences required by frozen feature-ready train/val/test; no labels used for embedding"}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(prepare(a.root, a.output, a.shards), indent=2))
