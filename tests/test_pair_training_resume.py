from pathlib import Path
from runpy import run_path
import json
import os
import signal
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
Sampler = run_path(str(ROOT / "scripts/training/train_esm_pair.py"))["PairTokenBatchSampler"]


def sampler():
    return Sampler([(4 + i, 7 + i % 3) for i in range(20)], 3, 1000, 10000, 10000, True, 49)


def test_resume_cursor_keeps_same_epoch_order_and_drops_no_remaining_pairs():
    full = sampler()
    full.epoch = 3
    expected = list(full)
    resumed = sampler()
    resumed.epoch = 3
    resumed.start_batch = 2
    assert list(resumed) == expected[2:]
    assert len(resumed) == len(expected)
    resumed.start_batch = 0
    assert list(resumed) == list(full)


def test_resume_at_epoch_end_yields_no_repeated_training_batches():
    resumed = sampler()
    resumed.start_batch = len(resumed)
    assert list(resumed) == []


def test_invalid_resume_cursor_fails_closed():
    resumed = sampler()
    resumed.start_batch = len(resumed) + 1
    with pytest.raises(ValueError, match="cursor"):
        list(resumed)


def test_full_training_has_no_smoke_caps_and_keeps_test_sealed():
    config = yaml.safe_load((ROOT / "configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml").read_text())
    train = config["training"]
    assert not {"max_train_pairs", "max_val_pairs", "smoke_max_chain_length"}.intersection(train)
    assert train["initialization"] == "scratch"
    assert train["evaluate_test"] is False
    assert train["checkpoint_every_steps"] > 0
    assert train["epochs"] == 15


@pytest.mark.parametrize("interrupt_phase", ["train", "val"])
def test_trainer_signal_resume_matches_uninterrupted_cpu_run(tmp_path, monkeypatch, interrupt_phase):
    fixture = run_path(str(ROOT / "tests/test_rcsb_pair_dataset.py"))
    paths = fixture["_prepare"](tmp_path)
    module = run_path(str(ROOT / "scripts/training/train_esm_pair.py"))
    scope = module["_main"].__globals__
    build_loader = scope["_build_loader"]
    # Synthetic fixture reuse tests continuation, not biological generalization.
    monkeypatch.setitem(scope, "_build_loader", lambda c, s, shuffle: build_loader(c, "train", shuffle))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = yaml.safe_load((ROOT / "configs/experiments/20261016_rcsb/rcsb_s5_noESM2contact_full.yaml").read_text())
    for key, value in zip(("pair_manifest", "pair_label_root", "rcsb_esm_feature_root", "rcsb_raw_ps_root", "rcsb_normalization_stats"), paths):
        config["data"][key] = str(value)
    config["model"].update(d_model=16, d_hidden=32)
    config["pair_model"].update(pair_rank=2, pair_hidden=4, bottleneck=8)
    config["training"].update(device="cpu", amp=False, epochs=2, batch_size=1,
                              eval_batch_size=1, num_workers=0, checkpoint_every_steps=1,
                              torch_num_threads=1)
    original_signals = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        config_path = tmp_path / "config.yaml"
        reference = tmp_path / "reference"
        resumed = tmp_path / "resumed"
        config["training"]["output_dir"] = str(reference)
        config_path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(sys, "argv", ["trainer", "--config", str(config_path)])
        assert scope["main"]() == 0

        config["training"]["output_dir"] = str(resumed)
        config_path.write_text(yaml.safe_dump(config))
        original_step = torch.amp.GradScaler.step
        original_evaluate = scope["_evaluate"]
        def stop_after_step(self, *args, **kwargs):
            result = original_step(self, *args, **kwargs)
            os.kill(os.getpid(), signal.SIGUSR1)
            return result
        def stop_during_validation(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGUSR1)
            return original_evaluate(*args, **kwargs)
        if interrupt_phase == "train":
            monkeypatch.setattr(torch.amp.GradScaler, "step", stop_after_step)
        else:
            monkeypatch.setitem(scope, "_evaluate", stop_during_validation)
        assert scope["main"]() == 75
        progress = torch.load(resumed / "progress.pt", weights_only=False)
        expected_batches = 1 if interrupt_phase == "train" else 3
        assert progress["completed_batches"] == progress["global_step"] == expected_batches
        monkeypatch.setattr(torch.amp.GradScaler, "step", original_step)
        monkeypatch.setitem(scope, "_evaluate", original_evaluate)
        monkeypatch.setattr(sys, "argv", ["trainer", "--config", str(config_path), "--resume"])
        assert scope["main"]() == 0
        a = torch.load(reference / "last.pt", weights_only=False)
        b = torch.load(resumed / "last.pt", weights_only=False)
        assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
        assert a["val"] == b["val"]
        final = json.loads((resumed / "final_metrics.json").read_text())
        assert final["test"] is None
    finally:
        for sig, handler in original_signals.items():
            signal.signal(sig, handler)
