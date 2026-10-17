"""Build an external-reference-free PPIS split from audited SI components."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with Path(path).open() as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def pdb_key(value):
    value = str(value).strip().lower()
    check(re.fullmatch(r"(?:[a-z0-9]{4}|pdb_[a-z0-9]{8})", value) is not None,
          "Invalid PDB identifier")
    return value


class UnionFind:
    def __init__(self, ids):
        self.parent = {u: u for u in ids}
        self.size = {u: 1 for u in ids}

    def find(self, uid):
        while self.parent[uid] != uid:
            self.parent[uid] = self.parent[self.parent[uid]]
            uid = self.parent[uid]
        return uid

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if (self.size[left], right) < (self.size[right], left):
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def allocate(blocks, ratios, seed):
    check(len(ratios) == 3 and all(math.isfinite(v) and v > 0 for v in ratios)
          and math.isclose(sum(ratios), 1.0), "Ratios must be positive and sum to one")
    ordered = sorted(blocks, key=lambda block: block[0])
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    n = sum(map(len, ordered))
    targets = dict(zip(SPLITS, (n * r for r in ratios)))
    counts, assignments = Counter(), {}
    for block in ordered:
        split = max(SPLITS, key=lambda s: targets[s] - counts[s])
        assignments.update((u, split) for u in block)
        counts[split] += len(block)
    check(all(counts[s] > 0 for s in SPLITS), "Whole components cannot populate all three splits")
    return assignments, counts, targets


def build(manifest, mapping, clusters, hits, graph_summary, output, ratios=(0.8, 0.1, 0.1), seed=42,
          min_length=50):
    output = Path(output)
    check(not output.exists(), "Use a fresh output directory; existing artifacts are immutable")
    sequences = {}
    for row in records(manifest):
        uid, seq, digest = row["unique_id"], row["sequence"], row["seq_sha256"]
        check(uid not in sequences, "Duplicate unique ID")
        check(hashlib.sha256(seq.encode()).hexdigest() == digest == uid[5:]
              and uid.startswith("useq_"), "Sequence hash/ID mismatch")
        check(len(seq) == int(row["n_residues"]) >= min_length, "Sequence length mismatch")
        sequences[uid] = row
    check(bool(sequences), "Empty manifest")
    graph_audit = json.loads(Path(graph_summary).read_text())
    check(graph_audit["sequences"] == len(sequences), "Audited graph universe differs")
    expected_hits = graph_audit["counts"]["directed_hits"]
    check(expected_hits > 0 and all(graph_audit["counts"].get(k, 0) == 0 for k in
          ("cross_component_hits", "identity_below_threshold", "coverage_below_threshold")),
          "Source graph audit did not pass")
    uf, members = UnionFind(sequences), {}
    with Path(clusters).open() as handle:
        for rep, uid in csv.reader(handle, delimiter="\t"):
            check(rep in sequences and uid in sequences and uid not in members,
                  "Invalid or duplicate cluster membership")
            members[uid] = rep
            uf.union(rep, uid)
    check(set(members) == set(sequences), "Cluster coverage mismatch")
    sequence_components = len({uf.find(u) for u in sequences})
    first_pdb, seen_chains, mapped, copies = {}, set(), set(), Counter()
    mapping_fields = None
    for row in records(mapping):
        mapping_fields = list(row)
        uid, chain, pdb = row["unique_id"], row["chain_instance_id"], pdb_key(row["pdb_id"])
        check(uid in sequences and chain not in seen_chains and bool(pdb), "Invalid chain mapping")
        check(row["seq_sha256"] == sequences[uid]["seq_sha256"], "Chain hash mismatch")
        check(not {"split", "component_id"} & row.keys(), "Input mapping already contains split columns")
        seen_chains.add(chain)
        mapped.add(uid)
        copies[uid] += 1
        if pdb in first_pdb:
            uf.union(first_pdb[pdb], uid)
        else:
            first_pdb[pdb] = uid
    check(mapped == set(sequences), "Chain mapping does not cover the sequence universe")
    groups = defaultdict(list)
    for uid in sorted(sequences):
        groups[uf.find(uid)].append(uid)
    blocks = list(groups.values())
    component = {u: block[0] for block in blocks for u in block}
    assignments, counts, targets = allocate(blocks, ratios, seed)

    # Audit the actual detected edges, not only cluster representatives.
    scanned = self_hits = 0
    self_ids = set()
    start = time.monotonic()
    hits_digest = hashlib.sha256()
    with Path(hits).open("rb") as handle:
        for line in handle:
            hits_digest.update(line)
            fields = line.decode().rstrip("\r\n").split("\t")
            check(len(fields) == 8, "Unexpected hit format")
            q, t, identity, alnlen, qc, tc, _, _ = fields
            check(q in assignments and t in assignments, "Hit outside manifest")
            check(members[q] == members[t], "Hit crosses original SI components")
            check(assignments[q] == assignments[t], "Detected SI edge crosses split")
            check(int(alnlen) > 0 and .3 - 1e-6 <= float(identity) <= 1.0,
                  "Invalid stored identity/alignment length")
            check(all(math.isfinite(float(c)) and .8 - 1e-3 <= float(c) <= 1.0 for c in (qc, tc)),
                  "Invalid hit coverage")
            scanned += 1
            self_hits += q == t
            if q == t:
                self_ids.add(q)
            if scanned % 10000000 == 0:
                print(json.dumps({"scanned_hits": scanned, "seconds": time.monotonic() - start}), flush=True)
    check(scanned == expected_hits, "Hit file row count differs from completed source audit")
    check(self_ids == set(sequences) and self_hits == len(sequences), "Self-hit coverage mismatch")

    pdb_splits, assembly_splits = {}, {}
    chain_counts, chain_residues = Counter(), Counter()
    for row in records(mapping):
        uid, pdb = row["unique_id"], pdb_key(row["pdb_id"])
        split = assignments[uid]
        assembly = (pdb, row["assembly_id"])
        check(pdb_splits.setdefault(pdb, split) == split, "PDB crosses split")
        check(assembly_splits.setdefault(assembly, split) == split, "Assembly crosses split")
        chain_counts[split] += 1
        chain_residues[split] += int(sequences[uid]["n_residues"])

    result = dict(
        status="complete_candidate", seed=seed, requested_ratios=dict(zip(SPLITS, ratios)),
        external_protection=False, external_exclusions=0,
        unique_sequences=len(sequences), chain_instances=len(seen_chains),
        pdbs=len(pdb_splits), assemblies=len(assembly_splits),
        sequence_components=sequence_components, joint_components=len(blocks),
        largest_components=sorted(map(len, blocks), reverse=True)[:20],
        directed_hits=scanned, self_hits=self_hits,
        unique_by_split=dict(counts), target_unique_by_split=targets,
        chain_instances_by_split=dict(chain_counts), chain_residues_by_split=dict(chain_residues),
        pdbs_by_split=dict(Counter(pdb_splits.values())),
        assemblies_by_split=dict(Counter(assembly_splits.values())),
        audit=dict(cross_detected_si_edges=0, cross_exact_sequences=0, cross_pdbs=0,
                   cross_assemblies=0, cross_joint_components=0),
        inputs={str(Path(p).resolve()): sha256(p) for p in (manifest, mapping, clusters, graph_summary)},
        hits_sha256=hits_digest.hexdigest(), script_sha256=sha256(__file__),
        scope="Candidate only: MMseqs detected SI30/bidirectional cov80 edges plus same-PDB grouping; no exhaustive global SI guarantee; pair and feature gates pending",
    )
    result["unique_length_by_split"] = {}
    for split in SPLITS:
        lengths = sorted(int(r["n_residues"]) for u, r in sequences.items() if assignments[u] == split)
        result["unique_length_by_split"][split] = dict(
            count=len(lengths), min=lengths[0], max=lengths[-1],
            mean=sum(lengths) / len(lengths), median=(lengths[(len(lengths)-1)//2]+lengths[len(lengths)//2])/2,
            below_100=sum(v < 100 for v in lengths), residues=sum(lengths))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "unique_split.tsv").open("x") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["unique_id", "split", "component_id", "length", "seq_sha256"])
        for uid in sorted(sequences):
            writer.writerow([uid, assignments[uid], component[uid], sequences[uid]["n_residues"],
                             sequences[uid]["seq_sha256"]])
    with (output / "chain_split.tsv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields + ["split", "component_id"], delimiter="\t")
        writer.writeheader()
        for row in records(mapping):
            uid = row["unique_id"]
            row["pdb_id"] = pdb_key(row["pdb_id"])
            writer.writerow(dict(row, split=assignments[uid], component_id=component[uid]))
    with (output / "components.tsv").open("x") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["component_id", "split", "unique_sequences", "chain_instances"])
        for block in sorted(blocks, key=lambda b: (-len(b), b[0])):
            writer.writerow([block[0], assignments[block[0]], len(block), sum(copies[u] for u in block)])
    with (output / "pdb_split.tsv").open("x") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["pdb_id", "split"])
        writer.writerows(sorted(pdb_splits.items()))
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "COMPLETED").touch(exist_ok=False)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "mapping", "clusters", "hits", "graph-summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=50)
    args = parser.parse_args()
    build(**vars(args))


if __name__ == "__main__":
    main()
