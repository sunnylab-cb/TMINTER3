"""Snapshot completed tag entries into exact-sequence and inverse manifests."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequences, mapped, pairs, entries = {}, [], 0, 0
    for path in sorted((args.tags / "entries").glob("*/*.json.gz")):
        with gzip.open(path, "rt") as handle:
            entry = json.load(handle)
        entries += 1
        used = {v for pair in entry["pairs"] for v in (pair["left"], pair["right"])}
        pairs += len(entry["pairs"])
        for instance in entry["chains"]:
            if instance["chain_instance_id"] not in used:
                continue
            source = entry["sources"][instance["source_label_asym_id"]]
            seq, digest = source["sequence"], source["seq_sha256"]
            assert hashlib.sha256(seq.encode()).hexdigest() == digest
            assert len(source["residue_map"]) == len(seq)
            if digest in sequences:
                assert sequences[digest] == seq
            sequences[digest] = seq
            mapped.append(dict(pdb_id=entry["pdb_id"], chain_instance_id=instance["chain_instance_id"],
                               assembly_id=instance["assembly_id"], source_label_asym_id=instance["source_label_asym_id"],
                               unique_id="useq_" + digest, seq_sha256=digest, tag_path=str(path.resolve())))
    with (args.output / "unique_sequences.tsv").open("x") as handle, (args.output / "unique_sequences.fasta").open("x") as fasta:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["unique_id", "seq_sha256", "n_residues", "sequence"])
        for digest, seq in sorted(sequences.items()):
            identifier = "useq_" + digest
            writer.writerow([identifier, digest, len(seq), seq])
            fasta.write(">{}\n{}\n".format(identifier, seq))
    with (args.output / "chain_to_unique.tsv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "chain_instance_id", "assembly_id", "source_label_asym_id", "unique_id", "seq_sha256", "tag_path"], delimiter="\t")
        writer.writeheader()
        writer.writerows(mapped)
    summary = dict(completed_entries=entries, interacting_pairs=pairs, chain_instances=len(mapped),
                   unique_sequences=len(sequences), residues=sum(map(len, sequences.values())),
                   scope="snapshot of completed entries; not final split or completeness assertion")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
