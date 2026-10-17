import csv
import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest
import torch

from ppcbind.data.esm_pair_dataset import collate_esm_pair_features
from ppcbind.data.rcsb_pair_dataset import (
    RCSB_NO_GRAPH_ABLATION,
    RCSB_PAIR_FEATURE_MODE,
    RCSBPairDataset,
    compute_train_uid_checksum,
)
from ppcbind.models.esm_pair import ESMPairSiteClassifier

_build_loader = run_path(str(Path(__file__).parents[1] / "scripts/training/train_esm_pair.py"))["_build_loader"]


MANIFEST_FIELDS = [
    "pair_id",
    "pdb_id",
    "assembly_id",
    "split",
    "left_seq_id",
    "right_seq_id",
    "left_length",
    "right_length",
    "left_positive",
    "right_positive",
    "left_observed",
    "right_observed",
    "n_contacts",
    "n_valid_pair_cells",
    "left_sequence_sha1",
    "right_sequence_sha1",
    "label_path",
    "label_offset",
]


def _uid(sequence: str) -> str:
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    return f"useq_{digest}"


def _write_features(root: Path, sequence: str, scalar_value: float = 2.0) -> str:
    uid = _uid(sequence)
    sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    prefix = uid[5:7]
    esm_path = root / "esm" / prefix / f"{uid}.pt"
    esm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequence": sequence,
            "sequence_sha256": sequence_hash,
            "embeddings": torch.zeros((len(sequence), 3456)),
            "repr_layers": [12, 24, 36],
            "hidden_normalization": "final_layernorm",
            "resolved_revision": "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c",
            "coordinates": torch.full((len(sequence), 3), 999.0),
        },
        esm_path,
    )
    raw_ps_path = root / "raw_ps" / "pt" / prefix / f"{uid}.pt"
    raw_ps_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequence": sequence,
            "sequence_sha256": sequence_hash,
            "physchem_features": torch.full((len(sequence), 62), 777.0),
            "spatial_scalar_raw_features": torch.full((len(sequence), 89), scalar_value),
            "spatial_vector_features": torch.zeros((len(sequence), 8, 3)),
            "source": "protenix_mini_esm_v0.5.0",
            "coordinates": torch.full((len(sequence), 3), 999.0),
        },
        raw_ps_path,
    )
    return uid


def _record(pair_id: str, split: str, left_uid: str, right_uid: str) -> dict:
    return {
        "pair_id": pair_id,
        "pdb_id": "1abc",
        "assembly_id": "1",
        "split": split,
        "left_seq_id": left_uid,
        "right_seq_id": right_uid,
        "left_observed_indices": [0, 1, 2],
        "right_observed_indices": [1, 2, 3],
        "left_positive": [0],
        "right_positive": [1],
        "residue_contacts": [[0, 1]],
        "contract": "observed-polymer-heavy-atom-5p5-assembly-v2",
    }


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    esm_root = tmp_path / "esm"
    raw_ps_root = tmp_path / "raw_ps"
    label_root = tmp_path / "labels"
    label_root.mkdir()
    train_left = _write_features(tmp_path, "ACDE")
    train_right = _write_features(tmp_path, "FGHI")
    short_left = _write_features(tmp_path, "AC")
    short_right = _write_features(tmp_path, "FG")
    test_left = _write_features(tmp_path, "KLMN")
    test_right = _write_features(tmp_path, "PQRS")
    short_record = {
        "pair_id": "train-short",
        "pdb_id": "2abc",
        "assembly_id": "1",
        "split": "train",
        "left_seq_id": short_left,
        "right_seq_id": short_right,
        "observed_sequence_only": True,
        "left_positive": [0],
        "right_positive": [1],
        "residue_contacts": [[0, 1]],
        "contract": "observed-polymer-heavy-atom-5p5-assembly-v2",
    }
    rows = []
    records = [
        _record("train-pair", "train", train_left, train_right),
        short_record,
        _record("train-homomer", "train", train_left, train_left),
        _record("test-pair", "test", test_left, test_right),
    ]
    sequence_by_uid = {
        train_left: "ACDE",
        train_right: "FGHI",
        short_left: "AC",
        short_right: "FG",
        test_left: "KLMN",
        test_right: "PQRS",
    }
    offset = 0
    with (label_root / "pairs.jsonl").open("wb") as handle:
        for record in records:
            encoded = (json.dumps(record) + "\n").encode("utf-8")
            handle.write(encoded)
            left_seq = sequence_by_uid[record["left_seq_id"]]
            right_seq = sequence_by_uid[record["right_seq_id"]]
            left_observed = len(left_seq) if record.get("observed_sequence_only") else 3
            right_observed = len(right_seq) if record.get("observed_sequence_only") else 3
            rows.append(
                {
                    "pair_id": record["pair_id"],
                    "pdb_id": record["pdb_id"],
                    "assembly_id": record["assembly_id"],
                    "split": record["split"],
                    "left_seq_id": record["left_seq_id"],
                    "right_seq_id": record["right_seq_id"],
                    "left_length": len(left_seq),
                    "right_length": len(right_seq),
                    "left_positive": 1,
                    "right_positive": 1,
                    "left_observed": left_observed,
                    "right_observed": right_observed,
                    "n_contacts": 1,
                    "n_valid_pair_cells": left_observed * right_observed,
                    "left_sequence_sha1": hashlib.sha1(left_seq.encode("ascii")).hexdigest(),
                    "right_sequence_sha1": hashlib.sha1(right_seq.encode("ascii")).hexdigest(),
                    "label_path": "pairs.jsonl",
                    "label_offset": offset,
                }
            )
            offset += len(encoded)
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "mean": [1.0] * 89,
                "std": [2.0] * 89,
                "feature_count": 89,
                "fit_split": "train",
                "train_uid_checksum": compute_train_uid_checksum(manifest),
                "train_unique_ids_sha256": compute_train_uid_checksum(manifest),
            }
        )
    )
    return manifest, label_root, esm_root, raw_ps_root, stats


def _dataset(paths: tuple[Path, Path, Path, Path, Path], **kwargs) -> RCSBPairDataset:
    manifest, label_root, esm_root, raw_ps_root, stats = paths
    options = {
        "pair_manifest": manifest,
        "pair_label_root": label_root,
        "split": "train",
        "esm_feature_root": esm_root,
        "raw_ps_root": raw_ps_root,
        "normalization_stats": stats,
        "feature_mode": RCSB_PAIR_FEATURE_MODE,
        "contact_graph_mode": RCSB_NO_GRAPH_ABLATION,
    }
    options.update(kwargs)
    return RCSBPairDataset(**options)


def test_adapter_emits_normalized_90d_features_and_existing_pair_batch(tmp_path: Path) -> None:
    dataset = _dataset(_prepare(tmp_path))
    assert len(dataset) == 3
    item = dataset[0]
    assert item["pair_id"] == "train-pair"
    assert item["left"]["seq_features"].shape == (4, 90)
    assert torch.equal(item["left"]["seq_features"][:, :89], torch.full((4, 89), 0.5))
    assert torch.equal(item["left"]["seq_features"][:, 89], torch.ones(4))
    assert item["left"]["labels"].tolist() == [1, 0, 0, -100]
    assert item["right"]["labels"].tolist() == [-100, 1, 0, 0]
    assert item["contact_index"].tolist() == [[0], [1]]
    assert not any("coordinate" in key for side in ("left", "right") for key in item[side])

    batch = collate_esm_pair_features([item])
    assert batch["left"]["seq_features"].shape == (1, 4, 90)
    assert batch["left"]["contact_edge_index"].shape == (2, 0)
    assert batch["right"]["aux_contact_edge_index"].shape == (2, 0)

    full_observation_item = dataset[1]
    assert full_observation_item["left_observed_mask"].tolist() == [True, True]
    assert full_observation_item["right_observed_mask"].tolist() == [True, True]
    assert full_observation_item["left"]["labels"].tolist() == [1, 0]


@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_protenix_vector_schema_is_validated(tmp_path: Path, fault: str) -> None:
    paths = _prepare(tmp_path)
    uid = _uid("ACDE")
    path = paths[3] / "pt" / uid[5:7] / f"{uid}.pt"
    payload = torch.load(path, weights_only=False)
    assert "vectors" not in payload
    if fault == "missing":
        # The old adapter name must not silently substitute for the real schema.
        payload["vectors"] = payload.pop("spatial_vector_features")
    elif fault == "shape":
        payload["spatial_vector_features"] = torch.zeros((4, 7, 3))
    else:
        payload["spatial_vector_features"][0, 0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="spatial_vector_features"):
        _dataset(paths)[0]


def test_homomer_cache_reads_keep_side_labels_independent(tmp_path: Path) -> None:
    dataset = _dataset(_prepare(tmp_path))

    item = dataset[2]

    assert item["left"] is not item["right"]
    assert item["left"]["labels"].tolist() == [1, 0, 0, -100]
    assert item["right"]["labels"].tolist() == [-100, 1, 0, 0]
    repeated = dataset[2]
    assert repeated["left"]["labels"].tolist() == [1, 0, 0, -100]
    assert repeated["right"]["labels"].tolist() == [-100, 1, 0, 0]


def test_feature_cache_hit_still_checks_expected_sha1(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    with paths[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    homomer_row = next(row for row in rows if row["pair_id"] == "train-homomer")
    homomer_row["right_sequence_sha1"] = "0" * 40
    with paths[0].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    stats = json.loads(paths[4].read_text())
    checksum = compute_train_uid_checksum(paths[0])
    stats["train_uid_checksum"] = checksum
    stats["train_unique_ids_sha256"] = checksum
    paths[4].write_text(json.dumps(stats))

    with pytest.raises(ValueError, match="manifest sequence SHA1 mismatch for cached UID"):
        _dataset(paths)[2]


@pytest.mark.parametrize(
    "contacts",
    [[], [[0, 1.5]], [[True, 1]], [[4, 1]]],
    ids=["empty", "fractional", "bool", "out-of-range"],
)
def test_residue_contacts_reject_empty_noninteger_and_out_of_range(
    tmp_path: Path, contacts: list
) -> None:
    paths = _prepare(tmp_path)
    label_path = paths[1] / "pairs.jsonl"
    records = [json.loads(line) for line in label_path.read_text().splitlines()]
    records[0]["residue_contacts"] = contacts
    label_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="residue_contacts"):
        _dataset(paths)[0]


def test_stats_are_bound_to_train_uids_and_split_is_filtered(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    dataset = _dataset(paths)
    assert [row["pair_id"] for row in dataset.rows] == ["train-pair", "train-short", "train-homomer"]

    stats = json.loads(paths[4].read_text())
    stats["train_uid_checksum"] = "0" * 64
    paths[4].write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="train UID checksum"):
        _dataset(paths)


def test_missing_per_residue_observation_indices_fail_closed(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    label_path = paths[1] / "pairs.jsonl"
    record = json.loads(label_path.read_text().splitlines()[0])
    del record["left_observed_indices"]
    label_path.write_text(json.dumps(record) + "\n")
    manifest = paths[0]
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["label_offset"] = "0"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    stats = json.loads(paths[4].read_text())
    stats["train_uid_checksum"] = compute_train_uid_checksum(manifest)
    paths[4].write_text(json.dumps(stats))

    with pytest.raises(ValueError, match="left_observed_indices"):
        _dataset(paths)[0]


def test_site_positive_projection_must_match_contacts(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    label_path = paths[1] / "pairs.jsonl"
    records = [json.loads(line) for line in label_path.read_text().splitlines()]
    records[0]["left_positive"] = [1]
    label_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="left site labels are not the projection"):
        _dataset(paths)[0]


def test_feature_sequence_must_match_manifest_uid(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    with paths[0].open(newline="") as handle:
        train_row = next(row for row in csv.DictReader(handle) if row["split"] == "train")
    uid = train_row["left_seq_id"]
    feature_path = paths[2] / uid[5:7] / f"{uid}.pt"
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    payload["sequence"] = "ACDF"
    torch.save(payload, feature_path)

    with pytest.raises(ValueError, match="StarESMC sequence SHA256"):
        _dataset(paths)[0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repr_layers", [24, 12, 36], "repr_layers"),
        ("resolved_revision", "0" * 40, "resolved_revision"),
    ],
)
def test_esm_metadata_must_match_pinned_feature_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    paths = _prepare(tmp_path)
    with paths[0].open(newline="") as handle:
        train_row = next(row for row in csv.DictReader(handle) if row["split"] == "train")
    uid = train_row["left_seq_id"]
    feature_path = paths[2] / uid[5:7] / f"{uid}.pt"
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, feature_path)

    with pytest.raises(ValueError, match=message):
        _dataset(paths)[0]


@pytest.mark.parametrize(("root_index", "field", "value", "message"), [
    (2, "embeddings", float("nan"), "StarESMC embeddings.*non-finite"),
    (3, "spatial_scalar_raw_features", float("inf"), "spatial_scalar_raw_features.*non-finite"),
])
def test_nonfinite_feature_values_fail_closed(
    tmp_path: Path, root_index: int, field: str, value: float, message: str
) -> None:
    paths = _prepare(tmp_path)
    with paths[0].open(newline="") as handle:
        train_row = next(row for row in csv.DictReader(handle) if row["split"] == "train")
    uid = train_row["left_seq_id"]
    feature_path = paths[2] / uid[5:7] / f"{uid}.pt"
    if root_index == 3:
        feature_path = paths[3] / "pt" / uid[5:7] / f"{uid}.pt"
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    payload[field][0, 0] = value
    torch.save(payload, feature_path)

    with pytest.raises(ValueError, match=message):
        _dataset(paths)[0]


def test_rcsb_loader_smoke_selection_and_tiny_model_forward(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    paths = _prepare(tmp_path)
    manifest, label_root, esm_root, raw_ps_root, stats = paths
    config = {
        "data": {
            "pair_data_backend": "rcsb_observed",
            "pair_manifest": manifest,
            "pair_label_root": label_root,
            "rcsb_esm_feature_root": esm_root,
            "rcsb_raw_ps_root": raw_ps_root,
            "rcsb_normalization_stats": stats,
            "rcsb_feature_mode": RCSB_PAIR_FEATURE_MODE,
            "contact_graph_mode": RCSB_NO_GRAPH_ABLATION,
        },
        "model": {
            "d_esm": 3456,
            "d_seq": 90,
            "d_model": 16,
            "d_hidden": 32,
            "n_heads": 4,
            "dropout": 0.0,
            "classifier_dropout": 0.0,
            "use_seq_features": True,
            "use_contact_graph": False,
        },
        "pair_model": {"mode": "independent"},
        "training": {
            "num_workers": 0,
            "batch_size": 1,
            "eval_batch_size": 1,
            "pin_memory": False,
            "smoke_max_chain_length": 3,
            "max_train_pairs": 1,
        },
    }
    dataset, loader = _build_loader(config, "train", False)
    assert [row["pair_id"] for row in dataset.rows] == ["train-short"]
    assert "selected=1 pair_ids=['train-short']" in capsys.readouterr().out

    batch = next(iter(loader))
    model = ESMPairSiteClassifier(config["model"], config["pair_model"]).eval()
    input_keys = {
        "esm_embeddings",
        "seq_features",
        "protein_mask",
        "chain_ids",
        "chain_rel_pos",
        "protein_rel_pos",
        "contact_edge_index",
        "contact_edge_scores",
        "aux_contact_edge_index",
        "aux_contact_edge_scores",
    }
    left = {key: value for key, value in batch["left"].items() if key in input_keys}
    right = {key: value for key, value in batch["right"].items() if key in input_keys}
    with torch.no_grad():
        output = model(left, right)
    assert output["left_logits"].shape == (1, 2, 2)
    assert output["right_logits"].shape == (1, 2, 2)

    for smoke_key in ("smoke_max_chain_length", "max_train_pairs", "max_val_pairs"):
        config["training"] = {"num_workers": 0, smoke_key: 1}
        with pytest.raises(ValueError, match="test loader cannot be built with smoke limits"):
            _build_loader(config, "test", False)


def test_no_graph_mode_is_an_explicit_choice(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    with pytest.raises(ValueError, match="no verified ESM2 contact graph"):
        _dataset(paths, contact_graph_mode="")
