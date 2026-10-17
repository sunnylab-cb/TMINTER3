import csv
import hashlib
import json
from pathlib import Path
from runpy import run_path
import sys

import pytest


MODULE = run_path(str(Path(__file__).resolve().parents[1] / "scripts/pull_rcsb_training_to_hpc3.py"))
SOURCE_FIXTURE = b"small-source-fixture\n"


@pytest.fixture(autouse=True)
def small_contract(monkeypatch):
    scope = MODULE["check_snapshot"].__globals__
    monkeypatch.setitem(scope, "EXPECTED_PAIR_COUNTS", {"train": 1})
    monkeypatch.setitem(scope, "EXPECTED_SOURCE_DIGEST", hashlib.sha256(SOURCE_FIXTURE).hexdigest())


def test_transfer_is_pull_only_and_non_destructive(tmp_path):
    for stage in MODULE["STAGES"]:
        cmd = MODULE["command"](tmp_path, stage, 0)
        assert cmd[-2].startswith("star:/data1/wholeset/RCSBprot/")
        assert cmd[-1].startswith(str(tmp_path))
        assert not any("delete" in arg or "remove-source" in arg or "inplace" in arg for arg in cmd)
        assert "--partial-dir=.rsync-partial" in cmd
        assert "--bwlimit=0" in cmd
        assert stage != "originalCIF" and not stage.endswith("runs")


def fixture(root):
    uid = "useq_" + "a" * 64
    ready = root / "training/ppis_ge50_v1/ready_v1"
    ready.mkdir(parents=True)
    (root / "_transfer").mkdir()
    (ready / "summary.json").write_text(json.dumps({"ready_pairs_by_split": {"train": 1},
        "input_manifest_sha256": hashlib.sha256(SOURCE_FIXTURE).hexdigest()}))
    with (ready / "ready_pair_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "left_seq_id", "right_seq_id", "label_path"])
        writer.writeheader()
        writer.writerow(dict(split="train", left_seq_id=uid, right_seq_id=uid, label_path="shards/000.jsonl"))
    label = root / "training/ppis_ge50_v1/pairs/shards/000.jsonl"
    label.parent.mkdir(parents=True)
    label.write_text("{}\n")
    (root / "training/ppis_ge50_v1/pairs/pair_manifest.csv").write_bytes(SOURCE_FIXTURE)
    for directory in ("TMINTER3-ESMC/features/unique", "protenix/features/complete_raw_v1/pt"):
        path = root / directory / "aa" / (uid + ".pt")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
    return uid


def test_complete_coverage_audit(tmp_path):
    fixture(tmp_path)
    result = MODULE["audit_ready"](tmp_path)
    assert result["unique_endpoints"] == 1
    assert result["missing_features"] == 0
    assert result["tensor_content_validation"] is False


def test_missing_feature_is_not_marked_complete(tmp_path):
    uid = fixture(tmp_path)
    (tmp_path / "TMINTER3-ESMC/features/unique/aa" / (uid + ".pt")).unlink()
    with pytest.raises(ValueError, match="missing/empty"):
        MODULE["audit_ready"](tmp_path)


def test_controller_records_complete_only_after_all_stages_and_audit(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pull", "--root", str(tmp_path)])
    calls = []
    def fake_transfer(cmd, **kwargs):
        calls.append(cmd)
        return 0
    monkeypatch.setattr(MODULE["subprocess"], "call", fake_transfer)
    MODULE["main"]()
    result = json.loads((tmp_path / "_transfer/COMPLETED.json").read_text())
    assert result["status"] == "complete"
    assert result["bwlimit_kib"] == 0
    assert set(result["completed_stages"]) == set(MODULE["STAGES"])
    assert len(calls) == len(MODULE["STAGES"])


def test_permanent_rsync_error_stops_and_has_no_completion_marker(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pull", "--root", str(tmp_path)])
    monkeypatch.setattr(MODULE["subprocess"], "call", lambda *a, **k: 23)
    with pytest.raises(RuntimeError, match="exit=23"):
        MODULE["main"]()
    assert not (tmp_path / "_transfer/COMPLETED.json").exists()
    state = json.loads((tmp_path / "_transfer/status.json").read_text())
    assert state["status"] == "failed"


def test_internally_consistent_wrong_snapshot_is_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "training/ppis_ge50_v1/ready_v1/summary.json"
    summary = json.loads(path.read_text())
    summary["ready_pairs_by_split"] = {"train": 2}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="frozen v1"):
        MODULE["audit_ready"](tmp_path)


def test_parent_manifest_hash_is_anchored(tmp_path):
    fixture(tmp_path)
    (tmp_path / "training/ppis_ge50_v1/pairs/pair_manifest.csv").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        MODULE["audit_ready"](tmp_path)


def test_structure_only_never_pulls_esmc_or_claims_full_completion(tmp_path, monkeypatch):
    uid = fixture(tmp_path)
    (tmp_path / "TMINTER3-ESMC/features/unique/aa" / (uid + ".pt")).unlink()
    monkeypatch.setattr(sys, "argv", ["pull", "--root", str(tmp_path), "--feature-scope", "structure-only"])
    calls = []
    def fake_transfer(cmd, **kwargs):
        calls.append(cmd)
        return 0
    monkeypatch.setattr(MODULE["subprocess"], "call", fake_transfer)
    MODULE["main"]()
    assert not (tmp_path / "_transfer/COMPLETED.json").exists()
    state = json.loads((tmp_path / "_transfer/STRUCTURE_COMPLETED.json").read_text())
    assert state["status"] == "structure_complete_esmc_deferred"
    assert state["audit"]["esmc_deferred"]
    assert not state["audit"]["esmc_checked"]
    assert len(calls) == len(MODULE["STAGES"]) - 1
    assert all(not cmd[-2].endswith("TMINTER3-ESMC/features/") for cmd in calls)
