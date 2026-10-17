import csv
import hashlib
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preparations/prepare_rcsb_training_features.py"
SPEC = importlib.util.spec_from_file_location("prepare_rcsb_training_features", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)

FIELDS = [
    "pair_id", "pdb_id", "assembly_id", "split", "left_seq_id", "right_seq_id",
    "left_chain_instance_id", "right_chain_instance_id", "left_length", "right_length",
    "left_positive", "right_positive", "left_observed", "right_observed", "n_contacts",
    "n_valid_pair_cells", "left_sequence_sha1", "right_sequence_sha1", "label_path", "label_offset",
]


def uid_for(sequence):
    return "useq_" + hashlib.sha256(sequence.encode()).hexdigest()


def write_feature(root, sequence, values, *, esmc=True, protenix=True, source=PREP.FEATURE_SOURCE):
    uid = uid_for(sequence)
    esmc_path, ps_path = PREP.feature_paths(root, uid)
    if esmc:
        esmc_path.parent.mkdir(parents=True, exist_ok=True)
        esmc_path.write_bytes(b"audited-esmc-placeholder")
    if protenix:
        ps_path.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.tensor(values, dtype=torch.float32).reshape(len(sequence), 89)
        torch.save({
            "sequence": sequence,
            "sequence_sha256": uid[5:],
            "source": source,
            "spatial_scalar_raw_features": tensor,
        }, ps_path)
    return uid


def pair_row(pair_id, split, left_uid, right_uid, left_length, right_length):
    return {
        "pair_id": pair_id,
        "pdb_id": "1abc",
        "assembly_id": "1",
        "split": split,
        "left_seq_id": left_uid,
        "right_seq_id": right_uid,
        "left_chain_instance_id": pair_id + ":A",
        "right_chain_instance_id": pair_id + ":B",
        "left_length": str(left_length),
        "right_length": str(right_length),
        "left_positive": "1",
        "right_positive": "1",
        "left_observed": str(left_length),
        "right_observed": str(right_length),
        "n_contacts": "1",
        "n_valid_pair_cells": str(left_length * right_length),
        "left_sequence_sha1": "a" * 40,
        "right_sequence_sha1": "b" * 40,
        "label_path": "labels/shard_000.jsonl",
        "label_offset": "0",
    }


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RCSBTrainingFeaturesTests(unittest.TestCase):
    def test_smoke_selection_is_bounded_and_order_independent(self):
        rows = [{"pair_id": f"pair_{index:03d}"} for index in range(300)]
        heap = []
        for ordinal, row in enumerate(reversed(rows)):
            PREP.offer_smoke(heap, row, ordinal, 256)
        self.assertEqual(
            [row["pair_id"] for row in PREP.selected_smoke_rows(heap)],
            [f"pair_{index:03d}" for index in range(256)],
        )

    def test_ready_filter_train_only_unique_residue_stats_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "features"
            output = base / "prepared"

            # Train values are [0, 2], [10], and [4, 6, 8], one value per residue.
            train_a = write_feature(root, "AA", [0] * 89 + [2] * 89)
            train_b = write_feature(root, "G", [10] * 89)
            train_c = write_feature(root, "VVV", [4] * 89 + [6] * 89 + [8] * 89)
            val_a = write_feature(root, "SS", [1000] * (2 * 89))
            val_b = write_feature(root, "ST", [1000] * (2 * 89))
            test_a = write_feature(root, "LL", [-1000] * (2 * 89))
            test_b = write_feature(root, "MPQ", [-1000] * (3 * 89))
            missing_esmc = uid_for("QQ")
            write_feature(root, "QQ", [12] * (2 * 89), esmc=False)
            empty_ps = uid_for("WW")
            write_feature(root, "WW", [], protenix=False)
            _, empty_ps_path = PREP.feature_paths(root, empty_ps)
            empty_ps_path.parent.mkdir(parents=True, exist_ok=True)
            empty_ps_path.touch()

            rows = [
                pair_row("pair_train_1", "train", train_a, train_b, 2, 1),
                pair_row("pair_train_2", "train", train_a, train_c, 2, 3),
                pair_row("pair_val", "val", val_a, val_b, 2, 2),
                pair_row("pair_test", "test", test_a, test_b, 2, 3),
                pair_row("pair_missing", "val", missing_esmc, empty_ps, 2, 2),
            ]
            manifest = base / "pairs.csv"
            write_manifest(manifest, rows)

            loaded_paths = []
            loaded_lock = threading.Lock()
            original_load = torch.load

            def tracked_load(path, *args, **kwargs):
                with loaded_lock:
                    loaded_paths.append(Path(path))
                return original_load(path, *args, **kwargs)

            with patch.object(torch, "load", side_effect=tracked_load):
                PREP.main([
                    "--pair-manifest", str(manifest),
                    "--feature-root", str(root),
                    "--output", str(output),
                    "--workers", "2",
                ])

            ready = read_csv(output / "ready_pair_manifest.csv")
            excluded = read_csv(output / "excluded_pairs.csv")
            self.assertEqual([row["pair_id"] for row in ready],
                             ["pair_train_1", "pair_train_2", "pair_val", "pair_test"])
            self.assertEqual(ready[2]["split"], "val")
            self.assertEqual(ready[3]["split"], "test")
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["pair_id"], "pair_missing")
            self.assertIn("missing_esmc", excluded[0]["left_reason"])
            self.assertIn("empty_protenix", excluded[0]["right_reason"])

            with (output / "missing_unique.tsv").open(encoding="utf-8") as handle:
                missing = {row["unique_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
            self.assertIn("missing_esmc", missing[missing_esmc]["reason"])
            self.assertIn("empty_protenix", missing[empty_ps]["reason"])

            stats = json.loads((output / "normalization_stats.json").read_text())
            self.assertEqual(stats["fit_split"], "train")
            self.assertEqual(stats["feature_count"], 89)
            self.assertEqual(stats["train_unique_count"], 3)
            self.assertEqual(stats["train_residue_count"], 6)
            self.assertAlmostEqual(stats["mean"][0], 5.0)
            self.assertAlmostEqual(stats["std"][0], (70.0 / 6.0) ** 0.5)
            self.assertTrue(all(abs(value - stats["mean"][0]) < 1e-12 for value in stats["mean"]))
            id_text = (output / "train_unique_ids.txt").read_text()
            expected_ids = "".join(uid + "\n" for uid in sorted((train_a, train_b, train_c)))
            self.assertEqual(id_text, expected_ids)
            self.assertEqual(stats["train_unique_ids_sha256"], hashlib.sha256(expected_ids.encode()).hexdigest())
            self.assertEqual(stats["train_uid_checksum"], stats["train_unique_ids_sha256"])
            self.assertEqual(stats["input_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertTrue(all("protenix" in str(path) for path in loaded_paths))
            self.assertEqual({path.stem for path in loaded_paths},
                             {train_a, train_b, train_c, val_a, val_b, test_a, test_b})
            self.assertEqual(len(loaded_paths), 7)

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["ready_pairs_by_split"], {"train": 2, "val": 1, "test": 1})
            self.assertEqual(summary["excluded_pairs_by_split"], {"train": 0, "val": 1, "test": 0})
            self.assertEqual(summary["heldout_protenix_validated_unique_endpoints_by_split"],
                             {"val": 2, "test": 2})
            self.assertEqual(summary["heldout_protenix_validated_residues_by_split"],
                             {"val": 4, "test": 5})
            self.assertIn("tensor content not validated", summary["feature_validation_scope"]["esmc"])
            self.assertEqual(len(read_csv(output / "smoke_train_pair_manifest.csv")), 2)
            self.assertEqual(len(read_csv(output / "smoke_val_pair_manifest.csv")), 1)
            self.assertNotIn("test_a", json.dumps(summary["smoke"]))

    def test_rejects_a_uid_assigned_to_multiple_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            uid_a, uid_b = uid_for("AA"), uid_for("GG")
            manifest = base / "pairs.csv"
            write_manifest(manifest, [
                pair_row("train_pair", "train", uid_a, uid_b, 2, 2),
                pair_row("val_pair", "val", uid_a, uid_b, 2, 2),
            ])
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                PREP.main([
                    "--pair-manifest", str(manifest),
                    "--feature-root", str(base / "features"),
                    "--output", str(base / "prepared"),
                ])
            self.assertFalse((base / "prepared").exists())

    def test_invalid_training_tensor_fails_instead_of_dropping_uid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "features"
            uid_a = write_feature(root, "AA", [float("nan")] * (2 * 89))
            uid_b = write_feature(root, "GG", [1] * (2 * 89))
            manifest = base / "pairs.csv"
            write_manifest(manifest, [pair_row("train_pair", "train", uid_a, uid_b, 2, 2)])
            with self.assertRaisesRegex(ValueError, "nonfinite values"):
                PREP.main([
                    "--pair-manifest", str(manifest),
                    "--feature-root", str(root),
                    "--output", str(base / "prepared"),
                ])
            self.assertFalse((base / "prepared").exists())

    def test_missing_test_split_is_reported_without_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "features"
            uid_a = write_feature(root, "AA", [0] * (2 * 89))
            uid_b = write_feature(root, "GG", [2] * (2 * 89))
            manifest = base / "pairs.csv"
            write_manifest(manifest, [pair_row("train_pair", "train", uid_a, uid_b, 2, 2)])
            output = base / "prepared"

            PREP.main([
                "--pair-manifest", str(manifest),
                "--feature-root", str(root),
                "--output", str(output),
            ])

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["source_pairs_by_split"], {"train": 1, "val": 0, "test": 0})
            self.assertEqual(summary["ready_pairs_by_split"], {"train": 1, "val": 0, "test": 0})
            self.assertEqual(summary["smoke"]["test_pairs_excluded"], 0)

    def test_invalid_heldout_tensor_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "features"
            train_a = write_feature(root, "AA", [0] * (2 * 89))
            train_b = write_feature(root, "GG", [2] * (2 * 89))
            val_a = write_feature(root, "SS", [1] * (2 * 89), source="unexpected_source")
            val_b = write_feature(root, "ST", [1] * (2 * 89))
            manifest = base / "pairs.csv"
            write_manifest(manifest, [
                pair_row("train_pair", "train", train_a, train_b, 2, 2),
                pair_row("val_pair", "val", val_a, val_b, 2, 2),
            ])
            output = base / "prepared"

            with self.assertRaisesRegex(ValueError, "unexpected Protenix source"):
                PREP.main([
                    "--pair-manifest", str(manifest),
                    "--feature-root", str(root),
                    "--output", str(output),
                ])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
