import gzip
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/preparations"))
from materialize_rcsb_ppis_pairs import CONTRACT, convert_entry, init_worker, shard_task


class PairTests(unittest.TestCase):
    def fixture(self, path):
        sources, chains, mapped, assignments = {}, [], [], {}
        for side, seq in (("A", "A" * 50), ("B", "G" * 55), ("C", "C" * 49)):
            digest = hashlib.sha256(seq.encode()).hexdigest()
            uid = "useq_" + digest
            sources[side] = dict(sequence=seq, seq_sha256=digest, residue_map=[{"index": i} for i in range(len(seq))])
            chains.append(dict(chain_instance_id="1abc:" + side, source_label_asym_id=side, assembly_id="1"))
            mapped.append(dict(pdb_id="1abc", chain_instance_id="1abc:" + side, source_label_asym_id=side,
                               assembly_id="1", seq_sha256=digest, unique_id=uid))
            if len(seq) >= 50:
                assignments[uid] = dict(split="train", length=str(len(seq)))
        pairs = [dict(left="1abc:A", right="1abc:" + s, left_positive=[1], right_positive=[2],
                      residue_contacts=[[1, 2]]) for s in ("B", "C")]
        entry = dict(contract=CONTRACT, pdb_id="1abc", sources=sources, chains=chains, pairs=pairs)
        with gzip.open(path, "wt") as handle:
            json.dump(entry, handle)
        init_worker(assignments, 50)
        return entry, mapped, assignments

    def test_short_partner_excluded_not_relabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1abc.json.gz"
            _, mapped, _ = self.fixture(path)
            rows, scanned, excluded = convert_entry(path, mapped)
            self.assertEqual((len(rows), scanned, excluded), (1, 2, 1))
            row, label = rows[0]
            self.assertEqual(row["n_contacts"], 1)
            self.assertEqual(row["left_positive"], 1)
            self.assertEqual(label["left_positive"], [1])
            self.assertEqual(label["residue_contacts"], [[1, 2]])
            self.assertEqual(row["n_valid_pair_cells"], 2750)

    def test_cross_split_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1abc.json.gz"
            _, mapped, assigned = self.fixture(path)
            assigned[mapped[1]["unique_id"]]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "endpoints cross split"):
                convert_entry(path, mapped)

    def test_written_byte_offsets_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "1abc.json.gz"
            _, mapped, _ = self.fixture(path)
            (root / "labels").mkdir()
            (root / "parts").mkdir()
            report = shard_task((0, [(str(path), mapped)], str(root)))
            self.assertEqual(report["pairs"], {"train": 1})
            with (root / "parts/000.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                with (root / row["label_path"]).open("rb") as handle:
                    handle.seek(int(row["label_offset"]))
                    label = json.loads(handle.readline())
                self.assertEqual(label["pair_id"], row["pair_id"])
                self.assertTrue(label["observed_sequence_only"])
                self.assertEqual(label["left_positive"], [1])

    def test_reversed_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1abc.json.gz"
            entry, mapped, _ = self.fixture(path)
            entry["pairs"].append(dict(left="1abc:B", right="1abc:A", left_positive=[2], right_positive=[1], residue_contacts=[[2, 1]]))
            with gzip.open(path, "wt") as handle:
                json.dump(entry, handle)
            with self.assertRaisesRegex(ValueError, "Duplicate/reversed"):
                convert_entry(path, mapped)


if __name__ == "__main__":
    unittest.main()
