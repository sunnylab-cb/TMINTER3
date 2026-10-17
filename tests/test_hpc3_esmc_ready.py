import csv
import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MOD = run_path(str(ROOT / "scripts/preparations/prepare_hpc3_esmc_ready.py"))


def fixture(root, monkeypatch):
    registry = root / "sequences/observed_full_success_v1/unique_sequences.tsv"
    registry.parent.mkdir(parents=True)
    rows = []
    for seq in ("ACDE", "ACDEF", "ACDEFG", "ACDEFGH", "ACDEFGHI", "ACDEFGHIK"):
        digest = hashlib.sha256(seq.encode()).hexdigest()
        rows.append(dict(unique_id="useq_" + digest, seq_sha256=digest, n_residues=len(seq), sequence=seq))
    with registry.open("w", newline="") as f:
        w = csv.DictWriter(f, MOD["FIELDS"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    manifest = root / "training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, ["split", "left_seq_id", "right_seq_id", "left_length", "right_length"])
        w.writeheader()
        for i, split in enumerate(("train", "val", "test")):
            a, b = rows[i * 2:i * 2 + 2]
            w.writerow(dict(split=split, left_seq_id=a["unique_id"], right_seq_id=b["unique_id"],
                            left_length=a["n_residues"], right_length=b["n_residues"]))
    receipt = root / "_transfer/metadata_snapshot.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"ready_sha256": MOD["sha_file"](manifest)}))
    scope = MOD["prepare"].__globals__
    monkeypatch.setitem(scope, "EXPECTED_PAIRS", {s: 1 for s in ("train", "val", "test")})
    monkeypatch.setitem(scope, "EXPECTED_UNIQUE", {s: 2 for s in ("train", "val", "test")})
    return rows, registry, manifest


def test_shards_partition_exact_sequences_without_trimming(tmp_path, monkeypatch):
    rows, _, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "inputs"
    result = MOD["prepare"](tmp_path, out, 4)
    assert result["total_unique"] == 6
    observed = []
    for part in result["shards"]:
        with (out / part["file"]).open() as f:
            observed.extend(csv.DictReader(f, delimiter="\t"))
    assert len({r["unique_id"] for r in observed}) == 6
    assert {r["sequence"] for r in observed} == {r["sequence"] for r in rows}
    assert result["layers"] == [12, 24, 36]
    assert result["embedding_dim"] == 3456


def test_changed_sequence_fails_hash_gate(tmp_path, monkeypatch):
    _, registry, _ = fixture(tmp_path, monkeypatch)
    registry.write_text(registry.read_text().replace("ACDE\n", "ACDX\n"))
    with pytest.raises(ValueError, match="mismatched"):
        MOD["prepare"](tmp_path, tmp_path / "inputs")


def test_changed_manifest_fails_snapshot_gate(tmp_path, monkeypatch):
    _, _, manifest = fixture(tmp_path, monkeypatch)
    with manifest.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="snapshot"):
        MOD["prepare"](tmp_path, tmp_path / "inputs")


def test_extractor_resume_rejects_wrong_model_revision(tmp_path):
    mod = run_path(str(ROOT / "scripts/preparations/extract_esmc_unique.py"))
    seq = "ACDE"
    sha = hashlib.sha256(seq.encode()).hexdigest()
    row = dict(sequence=seq, seq_sha256=sha, n_residues=4)
    payload = dict(sequence=seq, sequence_sha256=sha, repr_layers=[12, 24, 36],
                   hidden_normalization="final_layernorm", embeddings=torch.zeros(4, 3456),
                   resolved_revision="wrong")
    path = tmp_path / "feature.pt"
    torch.save(payload, path)
    revision = "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c"
    assert not mod["_valid_existing"](path, row, [12, 24, 36], 3456, "final_layernorm", revision)
    payload["resolved_revision"] = revision
    torch.save(payload, path)
    assert mod["_valid_existing"](path, row, [12, 24, 36], 3456, "final_layernorm", revision)
