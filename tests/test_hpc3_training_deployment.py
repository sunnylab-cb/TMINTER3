import copy
from pathlib import Path
import runpy
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/preparations/prepare_hpc3_training.py"))


def source_config():
    return yaml.safe_load((ROOT / "configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml").read_text())


def test_formal_contract_only_paths_change():
    source = source_config()
    full, smoke = BUILDER["rebase"](source, Path("/new"), Path("/deploy"))
    restored = copy.deepcopy(full)
    restored["data"] = source["data"]
    restored["training"]["output_dir"] = source["training"]["output_dir"]
    assert restored == source
    for key, value in full["data"].items():
        if key in ("pair_manifest", "pair_label_root", "rcsb_esm_feature_root",
                   "rcsb_raw_ps_root", "rcsb_normalization_stats"):
            assert value == source["data"][key].replace(BUILDER["OLD_ROOT"], "/new", 1)
        else:
            assert value == source["data"][key]
    assert "max_train_pairs" not in full["training"]
    assert smoke["training"]["max_train_pairs"] == 256
    assert smoke["training"]["max_val_pairs"] == 64
    assert not full["training"]["evaluate_test"]


@pytest.mark.parametrize("key,value", [("seed", 42), ("epochs", 1), ("evaluate_test", True),
                                      ("initialization", "warmstart"), ("max_train_pairs", 2)])
def test_invalid_formal_contract_rejected(key, value):
    source = source_config()
    source["training"][key] = value
    with pytest.raises(ValueError):
        BUILDER["rebase"](source, Path("/new"), Path("/deploy"))


def test_full_gate_fails_without_successful_smoke(tmp_path, monkeypatch):
    (tmp_path / "deployment.json").write_text('{"file_sha256": {}}')
    monkeypatch.setattr(sys, "argv", ["check", str(tmp_path), "full"])
    gate = runpy.run_path(str(ROOT / "scripts/preparations/check_hpc3_training.py"))
    with pytest.raises(FileNotFoundError):
        gate["main"]()
