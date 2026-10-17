import copy
import json
import random

import numpy as np
import pytest
import torch

from ppcbind import pair_checkpoint
from ppcbind.pair_checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    require_matching_config,
    restore_rng_state,
)


def _assert_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def _preserve_rng():
    state = capture_rng_state()
    yield
    restore_rng_state(state)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _training_state():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(),
        torch.nn.Dropout(0.4), torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scaler


def _train(state, steps):
    model, optimizer, scaler = state
    losses = []
    for _ in range(steps):
        inputs = torch.randn(5, 4) + random.random()
        targets = torch.from_numpy(np.random.randn(5, 2).astype(np.float32))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().clone())
    return losses


def test_interrupted_training_matches_uninterrupted_exactly(tmp_path):
    config = {
        "seed": 101, "optimizer": {"name": "Adam", "lr": 0.01},
        "features": ["sequence", "structure"], "pretrained": False,
        "metadata": {"split": "train", "checksum": "abc123", "optional": None},
    }
    _seed(101)
    uninterrupted = _training_state()
    expected_losses = _train(uninterrupted, 9)
    expected_rng = capture_rng_state()

    _seed(101)
    interrupted = _training_state()
    losses = _train(interrupted, 4)
    payload = {
        "model": interrupted[0].state_dict(),
        "optimizer": interrupted[1].state_dict(),
        "scaler": interrupted[2].state_dict(),
        "step": 4, "config": config, "rng": capture_rng_state(),
    }
    before_save = copy.deepcopy(payload)
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(payload, path)
    _assert_equal(before_save, payload)
    _assert_equal(payload["rng"], capture_rng_state())

    # Rebuilding a process consumes RNG; restoration must follow reconstruction.
    _seed(987)
    resumed = _training_state()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require_matching_config(checkpoint["config"], json.loads(json.dumps(config)))
    for component, key in zip(resumed, ("model", "optimizer", "scaler")):
        component.load_state_dict(checkpoint[key])
    restore_rng_state(checkpoint["rng"])
    losses += _train(resumed, 9 - checkpoint["step"])

    _assert_equal(losses, expected_losses)
    for actual, expected in zip(resumed, uninterrupted):
        _assert_equal(actual.state_dict(), expected.state_dict())
    _assert_equal(capture_rng_state(), expected_rng)


def test_capture_and_save_do_not_advance_rng(tmp_path):
    _seed(41)
    before = capture_rng_state()
    snapshot = capture_rng_state()
    _assert_equal(snapshot, before)
    atomic_torch_save({"rng": snapshot}, tmp_path / "rng.pt")
    _assert_equal(capture_rng_state(), before)
    expected = (random.random(), np.random.rand(3), torch.rand(3))
    restore_rng_state(snapshot)
    _assert_equal((random.random(), np.random.rand(3), torch.rand(3)), expected)


def test_atomic_save_replaces_using_complete_sibling(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1}, str(path))
    original_replace = pair_checkpoint.os.replace
    replacements = []

    def inspect_replace(source, destination):
        assert source.parent == path.parent
        assert source != path
        assert destination == path
        assert torch.load(path, weights_only=False) == {"step": 1}
        assert torch.load(source, weights_only=False) == {"step": 2}
        replacements.append(source)
        original_replace(source, destination)

    monkeypatch.setattr(pair_checkpoint.os, "replace", inspect_replace)
    atomic_torch_save({"step": 2}, path)
    assert len(replacements) == 1
    assert torch.load(path, weights_only=False) == {"step": 2}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["save", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_atomic_save_failure_cleans_temporary_file(tmp_path, monkeypatch, failure, existing):
    path = tmp_path / "checkpoint.pt"
    if existing:
        atomic_torch_save({"step": 1}, path)
    original_bytes = path.read_bytes() if existing else None

    def fail_save(payload, stream):
        stream.write(b"partial checkpoint")
        raise OSError("injected failure")

    def fail_replace(source, destination):
        raise OSError("injected failure")

    if failure == "save":
        monkeypatch.setattr(pair_checkpoint.torch, "save", fail_save)
    else:
        monkeypatch.setattr(pair_checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected failure"):
        atomic_torch_save({"step": 2}, path)
    if existing:
        assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


def test_matching_plain_metadata_dictionaries():
    saved = {"model": {"width": 8}, "splits": ["train", "test"], "amp": False, "tag": None}
    current = json.loads(json.dumps(saved))
    current = dict(reversed(list(current.items())))
    before = copy.deepcopy((saved, current))
    assert require_matching_config(saved, current) is None
    assert (saved, current) == before


@pytest.mark.parametrize("current", [
    {}, {"model": {"width": 9}}, {"model": {"width": 8}, "extra": None},
    {"model": {"width": "8"}}, None, [],
])
def test_mismatching_config_rejected(current):
    with pytest.raises(ValueError):
        require_matching_config({"model": {"width": 8}}, current)


def test_invalid_saved_config_rejected():
    with pytest.raises(ValueError):
        require_matching_config(None, {})


def test_ambiguous_config_equality_fails_closed():
    with pytest.raises(ValueError, match="cannot be compared"):
        require_matching_config({"x": np.array([1, 2])}, {"x": np.array([1, 2])})


def test_all_cuda_rng_states_captured_and_restored(monkeypatch):
    states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    snapshot = capture_rng_state()
    _assert_equal(snapshot["cuda"], states)
    restore_rng_state(snapshot)
    assert len(restored) == 1
    _assert_equal(restored[0], states)


@pytest.mark.parametrize("available,count", [(False, 0), (True, 1)])
def test_cuda_topology_mismatch_rejected(monkeypatch, available, count):
    state = capture_rng_state()
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8)] * 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    with pytest.raises(ValueError, match="same CUDA device count"):
        restore_rng_state(state)
