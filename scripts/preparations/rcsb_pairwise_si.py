"""Exhaustive sharded BioPython globalds SI graph on exact unique sequences.

Identity = exact matches / full gapped alignment length. Only a provable
composition/length upper bound skips pairs; no approximate-search prefilter.
This writes edges, not a finalized split. Complex constraints must also union.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import time
import warnings
from collections import Counter
from pathlib import Path

ROWS = None
COUNTS = None
MATRIX = None


def init(rows):
    global ROWS, COUNTS, MATRIX
    from Bio.Align import substitution_matrices
    ROWS = rows
    COUNTS = [Counter(r["sequence"]) for r in rows]
    MATRIX = substitution_matrices.load("BLOSUM62")


def compare_row(i):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from Bio import pairwise2
    left = ROWS[i]["sequence"]
    edges, skipped, aligned = [], 0, 0
    for j in range(i + 1, len(ROWS)):
        right = ROWS[j]["sequence"]
        bound = sum((COUNTS[i] & COUNTS[j]).values())
        if 10 * bound < 3 * max(len(left), len(right)):
            skipped += 1
            continue
        aln = pairwise2.align.globalds(left, right, MATRIX, -10, -0.5, one_alignment_only=True)[0]
        matches = sum(a == b for a, b in zip(aln.seqA, aln.seqB))
        length = len(aln.seqA)
        aligned += 1
        if 10 * matches >= 3 * length:
            edges.append((ROWS[i]["unique_id"], ROWS[j]["unique_id"], matches, length, matches / length))
    return i, edges, skipped, aligned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=120)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    assert 0 <= args.shard < args.shards
    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len({r["unique_id"] for r in rows}) == len(rows)
    assert len({r["sequence"] for r in rows}) == len(rows)
    for r in rows:
        assert hashlib.sha256(r["sequence"].encode()).hexdigest() == r["seq_sha256"]
        assert set(r["sequence"]) <= set("ARNDCQEGHILKMFPSTWYVBZX*")
    args.output.mkdir(parents=True, exist_ok=False)
    start, skipped, aligned, edge_count = time.monotonic(), 0, 0, 0
    with (args.output / "edges.tsv").open("x") as handle, mp.Pool(args.workers, initializer=init, initargs=(rows,)) as pool:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["left", "right", "matches", "alignment_length", "si"])
        for i, edges, sk, al in pool.imap_unordered(compare_row, range(args.shard, len(rows), args.shards), chunksize=1):
            writer.writerows(edges)
            handle.flush()
            skipped += sk
            aligned += al
            edge_count += len(edges)
            print(json.dumps(dict(row=i, aligned=aligned, proven_below_threshold=skipped, edges=edge_count, seconds=time.monotonic()-start)), flush=True)
    import Bio
    (args.output / "summary.json").write_text(json.dumps(dict(n_sequences=len(rows), shard=args.shard, shards=args.shards,
        aligned=aligned, proven_below_threshold=skipped, edges=edge_count, seconds=time.monotonic()-start,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), biopython=Bio.__version__,
        contract="globalds BLOSUM62 -10/-0.5 first alignment; matches/full gapped length >=0.30"), indent=2))


if __name__ == "__main__":
    main()
