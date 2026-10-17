"""CPU-only toy checks for external-free sequence/complex split boundaries."""
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/preparations"))
from partition_ppis_components import allocate, build


def table(path, header, rows):
    with path.open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if header:
            writer.writerow(header)
        writer.writerows(rows)


class SplitTests(unittest.TestCase):
    def fixture(self, root):
        seqs = ["A" * 50 + c for c in "CDEFGHIKLMNPQRSTVWY"]
        ids = ["useq_" + hashlib.sha256(s.encode()).hexdigest() for s in seqs]
        table(root / "manifest.tsv", ["unique_id", "seq_sha256", "n_residues", "sequence"],
              [[u, u[5:], len(s), s] for u, s in zip(ids, seqs)])
        rows = [[f"p{i:03d}", f"p{i:03d}_A", "1", "A", u, u[5:], f"/{i}.json.gz"]
                for i, u in enumerate(ids)]
        # A transitive SI edge and same-PDB link must merge all three IDs.
        rows[2][0] = rows[1][0]
        table(root / "mapping.tsv", ["pdb_id", "chain_instance_id", "assembly_id",
              "source_label_asym_id", "unique_id", "seq_sha256", "tag_path"], rows)
        clusters = [[ids[0] if i == 1 else u, u] for i, u in enumerate(ids)]
        table(root / "clusters.tsv", None, clusters)
        hits = [[u, u, 1, 51, 1, 1, 0, 100] for u in ids]
        hits.append([ids[0], ids[1], .4, 51, 1, 1, 1e-5, 30])
        table(root / "hits.tsv", None, hits)
        (root / "graph.json").write_text(json.dumps({"sequences": len(ids), "counts": {"directed_hits": len(hits)}}))
        return ids

    def run_build(self, root, output="out"):
        return build(root / "manifest.tsv", root / "mapping.tsv", root / "clusters.tsv",
                     root / "hits.tsv", root / "graph.json", root / output, ratios=(.6, .2, .2))

    def test_full_membership_pdb_and_si(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = self.fixture(root)
            result = self.run_build(root)
            with (root / "out/unique_split.tsv").open() as handle:
                assigned = {r["unique_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
            self.assertEqual(set(assigned), set(ids))
            self.assertEqual(len({assigned[u]["split"] for u in ids[:3]}), 1)
            self.assertEqual(len({assigned[u]["component_id"] for u in ids[:3]}), 1)
            self.assertEqual(result["external_exclusions"], 0)
            self.assertEqual(result["unique_sequences"], len(ids))
            self.assertEqual(sum(result["unique_by_split"].values()), len(ids))
            self.assertTrue((root / "out/COMPLETED").exists())
            with self.assertRaises(ValueError):
                self.run_build(root)
            self.run_build(root, "replica")
            self.assertEqual((root / "out/unique_split.tsv").read_bytes(),
                             (root / "replica/unique_split.tsv").read_bytes())

    def test_unrepresented_si_edge_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = self.fixture(root)
            with (root / "hits.tsv").open("a") as handle:
                handle.write(f"{ids[0]}\t{ids[-1]}\t.9\t51\t1\t1\t0\t100\n")
            with self.assertRaisesRegex(ValueError, "original SI components"):
                self.run_build(root)
            self.assertFalse((root / "out").exists())

    def test_duplicate_chain_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            lines = (root / "mapping.tsv").read_text().splitlines()
            with (root / "mapping.tsv").open("a") as handle:
                handle.write(lines[1] + "\n")
            with self.assertRaisesRegex(ValueError, "chain mapping"):
                self.run_build(root)

    def test_truncated_hits_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            lines = (root / "hits.tsv").read_text().splitlines()
            (root / "hits.tsv").write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "row count"):
                self.run_build(root)
            self.assertFalse((root / "out").exists())

    def test_pdb_whitespace_is_canonicalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / "mapping.tsv"
            lines = path.read_text().splitlines()
            fields = lines[3].split("\t")
            fields[0] = "  " + fields[0].upper() + " "
            lines[3] = "\t".join(fields)
            path.write_text("\n".join(lines) + "\n")
            result = self.run_build(root)
            self.assertEqual(result["pdbs"], 18)
            self.assertEqual(result["joint_components"], 17)

    def test_bad_ratios(self):
        for ratios in [(1, 0, 0), (.8, .2, .2), (float("nan"), .1, .1)]:
            with self.assertRaises(ValueError):
                allocate([["a"], ["b"], ["c"]], ratios, 42)


if __name__ == "__main__":
    unittest.main()
