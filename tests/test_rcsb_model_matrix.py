from pathlib import Path
from runpy import run_path
import signal
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml"
BUILD = run_path(str(ROOT / "scripts/preparations/build_rcsb_model_matrix.py"))["build_matrix"]


def test_matrix_changes_only_declared_contract_fields(tmp_path):
    base = yaml.safe_load(BASE.read_text())
    manifest = BUILD(BASE, tmp_path / "matrix")
    outputs = set()
    for row in manifest["variants"]:
        config = yaml.safe_load((tmp_path / "matrix" / row["config"]).read_text())
        assert config["data"] == base["data"]
        assert config["model"] == base["model"]
        train = dict(config["training"])
        outputs.add(train.pop("output_dir"))
        expected = dict(base["training"])
        expected.pop("output_dir")
        if row["name"] == "outer_contact":
            expected["pair_contact_loss_weight"] = 0.1
        assert train == expected
    assert len(outputs) == 4
    with pytest.raises(FileExistsError):
        BUILD(BASE, tmp_path / "matrix")


@pytest.mark.parametrize("variant_index", range(4))
def test_each_variant_trains_and_evaluates_on_real_adapter_fixture(tmp_path, monkeypatch, variant_index):
    paths = run_path(str(ROOT / "tests/test_rcsb_pair_dataset.py"))["_prepare"](tmp_path)
    manifest = BUILD(BASE, tmp_path / "matrix")
    row = manifest["variants"][variant_index]
    config = yaml.safe_load((tmp_path / "matrix" / row["config"]).read_text())
    for key, value in zip(("pair_manifest", "pair_label_root", "rcsb_esm_feature_root", "rcsb_raw_ps_root", "rcsb_normalization_stats"), paths):
        config["data"][key] = str(value)
    # Retain actual production model widths; only shrink the fixture and duration.
    config["training"].update(output_dir=str(tmp_path / "run"), device="cpu", amp=False,
                              epochs=1, batch_size=2, eval_batch_size=2, num_workers=0,
                              torch_num_threads=1)
    module = run_path(str(ROOT / "scripts/training/train_esm_pair.py"))
    scope = module["_main"].__globals__
    build_loader = scope["_build_loader"]
    # Reuse synthetic fixture for val solely to verify execution, not accuracy.
    monkeypatch.setitem(scope, "_build_loader", lambda c, s, shuffle: build_loader(c, "train", shuffle))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["trainer", "--config", str(config_path)])
    handlers = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        assert scope["main"]() == 0
        checkpoint = torch.load(tmp_path / "run/last.pt", weights_only=False)
        assert all(torch.isfinite(t).all() for t in checkpoint["model"].values())
        if row["name"] == "outer_contact":
            import json
            epoch = json.loads((tmp_path / "run/metrics.jsonl").read_text().splitlines()[0])
            assert epoch["train_pair_contact_loss"] > 0
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
