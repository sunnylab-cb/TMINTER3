"""Materialize observed partner-specific labels for a frozen sequence/PDB split."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from partition_ppis_components import check, pdb_key, records, sha256

CONTRACT = "observed-polymer-heavy-atom-5p5-assembly-v2"
FIELDS = ["pair_id", "pdb_id", "assembly_id", "split", "left_seq_id", "right_seq_id",
          "left_chain_instance_id", "right_chain_instance_id", "left_length", "right_length",
          "left_positive", "right_positive", "left_observed", "right_observed", "n_contacts",
          "n_valid_pair_cells", "left_sequence_sha1", "right_sequence_sha1", "label_path", "label_offset"]
ASSIGNMENTS = {}
MIN_LENGTH = 50


def init_worker(assignments, min_length):
    global ASSIGNMENTS, MIN_LENGTH
    ASSIGNMENTS, MIN_LENGTH = assignments, min_length


def convert_entry(path, expected):
    with gzip.open(path, "rt") as handle:
        entry = json.load(handle)
    check(entry["contract"] == CONTRACT, "Unexpected label contract")
    pdb = pdb_key(entry["pdb_id"])
    chains = {c["chain_instance_id"]: c for c in entry["chains"]}
    check(len(chains) == len(entry["chains"]), "Duplicate source chain instance")
    mapped = {r["chain_instance_id"]: r for r in expected}
    check(len(mapped) == len(expected), "Duplicate mapped chain instance")
    metadata = {}
    for chain_id, row in mapped.items():
        c = chains[chain_id]
        check(pdb_key(row["pdb_id"]) == pdb and c["source_label_asym_id"] == row["source_label_asym_id"]
              and str(c["assembly_id"]) == row["assembly_id"], "Chain provenance mismatch")
        source = entry["sources"][c["source_label_asym_id"]]
        seq = source["sequence"]
        digest = hashlib.sha256(seq.encode()).hexdigest()
        check(digest == source["seq_sha256"] == row["seq_sha256"]
              and row["unique_id"] == "useq_" + digest, "Sequence hash mismatch")
        check([r["index"] for r in source["residue_map"]] == list(range(len(seq))), "Residue index mismatch")
        uid = row["unique_id"]
        if len(seq) >= MIN_LENGTH:
            check(uid in ASSIGNMENTS and int(ASSIGNMENTS[uid]["length"]) == len(seq),
                  "Eligible sequence absent from split or wrong length")
        else:
            check(uid not in ASSIGNMENTS, "Short sequence entered split")
        metadata[chain_id] = (uid, len(seq), hashlib.sha1(seq.encode()).hexdigest(), str(c["assembly_id"]))
    seen, used = set(), set()
    result, excluded = [], 0
    for pair in entry["pairs"]:
        left, right = pair["left"], pair["right"]
        key = tuple(sorted((left, right)))
        check(left != right and key not in seen, "Duplicate/reversed/self chain-instance pair")
        check(left in metadata and right in metadata, "Unmapped pair endpoint")
        seen.add(key)
        used.update(key)
        lu, ll, lh, la = metadata[left]
        ru, rl, rh, ra = metadata[right]
        check(la == ra, "Pair spans biological assemblies")
        contacts = pair["residue_contacts"]
        check(bool(contacts) and all(len(c) == 2 and all(type(v) is int for v in c)
              and 0 <= c[0] < ll and 0 <= c[1] < rl for c in contacts), "Invalid contact indices")
        check(len({tuple(c) for c in contacts}) == len(contacts), "Duplicate residue contacts")
        lp, rp = sorted({c[0] for c in contacts}), sorted({c[1] for c in contacts})
        check(lp == pair["left_positive"] and rp == pair["right_positive"], "Contact projection mismatch")
        if min(ll, rl) < MIN_LENGTH:
            excluded += 1
            continue
        split = ASSIGNMENTS[lu]["split"]
        check(split == ASSIGNMENTS[ru]["split"], "Pair endpoints cross split")
        pair_id = "pair_" + hashlib.sha256((pdb + "|" + "|".join(key)).encode()).hexdigest()
        row = dict(pair_id=pair_id, pdb_id=pdb, assembly_id=la, split=split,
                   left_seq_id=lu, right_seq_id=ru, left_chain_instance_id=left,
                   right_chain_instance_id=right, left_length=ll, right_length=rl,
                   left_positive=len(lp), right_positive=len(rp), left_observed=ll,
                   right_observed=rl, n_contacts=len(contacts), n_valid_pair_cells=ll * rl,
                   left_sequence_sha1=lh, right_sequence_sha1=rh)
        label = dict(row, contract=CONTRACT, cutoff_angstrom=5.5, heavy_only=True,
                     observed_sequence_only=True,
                     residue_contacts=contacts)
        label["left_positive"], label["right_positive"] = lp, rp
        result.append((row, label))
    check(used == set(mapped), "Source chain mapping not exactly the interacting endpoints")
    return result, len(entry["pairs"]), excluded


def shard_task(task):
    index, paths, output = task
    output = Path(output)
    relative = f"labels/shard_{index:03d}.jsonl"
    counts, positive, residues = Counter(), Counter(), Counter()
    used, chains, pdbs = defaultdict(set), defaultdict(set), defaultdict(set)
    scanned = excluded = 0
    with (output / relative).open("xb") as labels, (output / f"parts/{index:03d}.csv").open("x") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=FIELDS)
        writer.writeheader()
        for path, expected in paths:
            converted, total, dropped = convert_entry(path, expected)
            scanned += total
            excluded += dropped
            for row, label in converted:
                row["label_path"], row["label_offset"] = relative, labels.tell()
                labels.write((json.dumps(label, separators=(",", ":")) + "\n").encode())
                writer.writerow(row)
                split = row["split"]
                counts[split] += 1
                positive[split] += row["left_positive"] + row["right_positive"]
                residues[split] += row["left_length"] + row["right_length"]
                used[split].update((row["left_seq_id"], row["right_seq_id"]))
                chains[split].update((row["left_chain_instance_id"], row["right_chain_instance_id"]))
                pdbs[split].add(row["pdb_id"])
    return dict(shard=index, source_pairs=scanned, excluded_short_partner=excluded,
                pairs=dict(counts), positive=dict(positive), residues=dict(residues),
                used={k: sorted(v) for k, v in used.items()}, chains={k: sorted(v) for k, v in chains.items()},
                pdbs={k: sorted(v) for k, v in pdbs.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--expected-split-sha256", required=True)
    p.add_argument("--full-mapping", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shards", type=int, default=64)
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument("--expected-source-pairs", type=int, required=True)
    a = p.parse_args()
    check(a.workers > 0 and a.shards > 0, "Invalid parallelism")
    check(a.min_length == 50, "This frozen cohort requires min_length=50")
    check((a.split_dir / "COMPLETED").is_file(), "Split audit incomplete")
    split_hash = sha256(a.split_dir / "unique_split.tsv")
    check(split_hash == a.expected_split_sha256, "Frozen split digest mismatch")
    assignments = {}
    for row in records(a.split_dir / "unique_split.tsv"):
        check(row["unique_id"] not in assignments and row["split"] in ("train", "val", "test"),
              "Duplicate UID or invalid split name")
        assignments[row["unique_id"]] = row
    split_summary = json.loads((a.split_dir / "summary.json").read_text())
    check(len(assignments) == split_summary["unique_sequences"], "Split size mismatch")
    check(dict(Counter(r["split"] for r in assignments.values())) == split_summary["unique_by_split"],
          "Split counts disagree with completed audit")
    check(not split_summary["external_protection"] and split_summary["external_exclusions"] == 0,
          "External exclusions forbidden for this PPIS run")
    groups, seen_pdbs, seen_chains = defaultdict(list), {}, set()
    for row in records(a.full_mapping):
        pdb, path = pdb_key(row["pdb_id"]), row["tag_path"]
        check(seen_pdbs.setdefault(pdb, path) == path, "Multiple source files for one PDB")
        check(row["chain_instance_id"] not in seen_chains, "Duplicate chain instance in full mapping")
        seen_chains.add(row["chain_instance_id"])
        groups[path].append(row)
    paths = sorted(groups.items())
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "labels").mkdir()
    (a.output / "parts").mkdir()
    tasks = [(i, paths[i::a.shards], str(a.output)) for i in range(a.shards)]
    counts, positive, residues = Counter(), Counter(), Counter()
    used, chains, pdbs = defaultdict(set), defaultdict(set), defaultdict(set)
    scanned = excluded = 0
    with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(assignments, a.min_length)) as pool:
        for done, result in enumerate(pool.map(shard_task, tasks), 1):
            scanned += result["source_pairs"]
            excluded += result["excluded_short_partner"]
            counts.update(result["pairs"])
            positive.update(result["positive"])
            residues.update(result["residues"])
            for key, dest in (("used", used), ("chains", chains), ("pdbs", pdbs)):
                for split, ids in result[key].items():
                    dest[split].update(ids)
            print(json.dumps(dict(shards_complete=done, source_pairs=scanned, selected_pairs=dict(counts))), flush=True)
    check(scanned == a.expected_source_pairs, "Full label census does not match frozen source pair count")
    check(scanned == sum(counts.values()) + excluded, "Pair accounting mismatch")
    for dest in (used, chains, pdbs):
        for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
            check(not dest[x] & dest[y], "Pair subset crosses split boundaries")
    with (a.output / "pair_manifest.csv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(a.shards):
            with (a.output / f"parts/{index:03d}.csv").open() as part:
                writer.writerows(csv.DictReader(part))
    for split, ids in used.items():
        (a.output / f"{split}_unique_ids.txt").write_text("".join(u + "\n" for u in sorted(ids)))
    summary = dict(status="complete", contract=CONTRACT, source_pairs=scanned, min_length=a.min_length,
                   excluded_short_partner=excluded, pairs_by_split=dict(counts),
                   unique_by_split={k: len(v) for k, v in used.items()},
                   chain_instances_by_split={k: len(v) for k, v in chains.items()},
                   pdbs_by_split={k: len(v) for k, v in pdbs.items()},
                   pair_weighted_residues=dict(residues), pair_weighted_positive=dict(positive),
                   unused_split_unique=len(assignments) - sum(map(len, used.values())),
                   split_summary_sha256=sha256(a.split_dir / "summary.json"),
                   unique_split_sha256=split_hash,
                   full_mapping_sha256=sha256(a.full_mapping), script_sha256=sha256(__file__),
                   scope="Undirected contacting chain-pair contexts, both observed chains >=min_length; two directional labels per record; no feature readiness filtering")
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.output / "COMPLETED").touch(exist_ok=False)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
