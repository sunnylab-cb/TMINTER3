import csv
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'objective_audit', ROOT / 'scripts/analysis/audit_pair_objective.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_coefficients_track_batch_mean_not_instance_mean():
    mass, sizes, batches, histogram = AUDIT.batch_coefficients([[0, 1], [2]], 3)
    np.testing.assert_allclose(mass, [.25, .25, .5])
    assert batches == 2 and histogram == {1: 1, 2: 1}
    assert list(sizes) == [2, 2, 1]


def test_incomplete_or_duplicate_sampler_rejected():
    with pytest.raises(ValueError):
        AUDIT.batch_coefficients([[0, 1], [1, 2]], 3)
    with pytest.raises(ValueError):
        AUDIT.batch_coefficients([[0, 1]], 3)


def test_weighted_smoothed_stationary_point():
    _, loss_class, _ = AUDIT.load_trainer_classes(ROOT / 'scripts/training/train_esm_pair.py')
    p0 = AUDIT.target_probability(0, 10.380668383505242, .05)
    p1 = AUDIT.target_probability(1, 10.380668383505242, .05)
    assert .210 < p0 < .211
    p = torch.tensor([[p0, p0, p1]], dtype=torch.float64)
    logits = torch.stack((torch.zeros_like(p), torch.logit(p)), -1).requires_grad_()
    loss = loss_class(10.380668383505242, .05, 0)._one_side(
        logits, torch.tensor([[0, 0, 1]]))
    loss.backward()
    assert logits.grad.abs().max().item() < 2e-6


def test_manifest_read_ignores_test_labels():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'manifest.csv'
        row = dict(pair_id='train_pair', split='train', left_seq_id='A', right_seq_id='B',
                   left_length=100, right_length=200, left_positive=1, right_positive=2,
                   left_observed=100, right_observed=200)
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            writer.writerow(dict(row, split='test', left_length='do_not_parse_test'))
        rows, digest = AUDIT.read_manifest(path, 'train')
        assert rows.tolist() == [[100, 200, 1, 2, 0]]
        assert len(digest) == 64


def test_empty_uid_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'manifest.csv'
        row = dict(pair_id='p', split='train', left_seq_id='', right_seq_id='B')
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        with pytest.raises(ValueError, match='sequence UID'):
            AUDIT.read_manifest(path, 'train')


def test_profile_matches_class_counts_and_weight():
    sampler, _, _ = AUDIT.load_trainer_classes(ROOT/'scripts/training/train_esm_pair.py')
    rows = np.array([[100, 100, 1, 2, 0]])
    snapshot = dict(config=dict(training=dict(batch_size=1, eval_batch_size=1,
        max_batch_tokens=1000, max_pair_cells=10000, max_single_pair_cells=10000,
        seed=49, label_smoothing=.05, ratio_loss_weight=.25), pair_model=dict(mode='outer_product')),
        n_train_pairs=1, train_batches_per_epoch=1, train_positive=3,
        train_negative=197, pos_weight=20.0)
    result = AUDIT.summarize_profile(rows, snapshot, sampler, 'train')
    assert result['batches'] == 1
    snapshot['train_positive'] = 4
    with pytest.raises(ValueError, match='class counts'):
        AUDIT.summarize_profile(rows, snapshot, sampler, 'train')
    snapshot['train_positive'] = 3
    snapshot['pos_weight'] = 2
    with pytest.raises(ValueError, match='pos_weight'):
        AUDIT.summarize_profile(rows, snapshot, sampler, 'train')


def test_hard_label_mean_normalization_is_not_soft_target_mean():
    torch.set_num_threads(1)
    logits = torch.tensor([[0.0, -.5], [0.0, .7]], dtype=torch.float64)
    weights = torch.tensor([1.0, 10.0], dtype=torch.float64)
    y1, y2 = torch.tensor([0, 0]), torch.tensor([1, 1])
    hard = sum(torch.nn.functional.cross_entropy(
        logits, labels, weight=weights) for labels in (y1, y2))/2
    soft = torch.nn.functional.cross_entropy(
        logits, torch.full_like(logits, .5), weight=weights)
    assert abs(hard.item() - soft.item()) > .1


def test_multicontext_formula_matches_frozen_loss_and_gradients():
    _, loss_class, _ = AUDIT.load_trainer_classes(ROOT / 'scripts/training/train_esm_pair.py')
    weight, smoothing, length = 10.380668383505242, .05, 100
    p = torch.linspace(.03, .97, length, dtype=torch.float64)
    logits = torch.stack((torch.zeros_like(p), torch.logit(p)), -1)[None].requires_grad_()
    labels = [(torch.arange(length) < k).long()[None] for k in (1, 50)]
    criterion = loss_class(weight, smoothing, 0)
    actual = sum(criterion._one_side(logits, y) for y in labels)/2
    denominators = torch.tensor(
        [length + (weight - 1)*int(y.sum()) for y in labels], dtype=torch.float64)
    targets = torch.stack([y[0] for y in labels]).double()*(1 - smoothing) + smoothing/2
    positive = (weight*targets/denominators[:, None]).mean(0)
    negative = ((1 - targets)/denominators[:, None]).mean(0)
    margin = logits[0, :, 1] - logits[0, :, 0]
    manual = -(positive*torch.nn.functional.logsigmoid(margin)
               + negative*torch.nn.functional.logsigmoid(-margin)).sum()
    grad_actual = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    grad_manual = torch.autograd.grad(manual, logits)[0]
    torch.testing.assert_close(actual.double(), manual, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(grad_actual, grad_manual, atol=2e-7, rtol=1e-6)


def test_sha1_groups_exact_sequences_not_chain_names():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'manifest.csv'
        row = dict(pair_id='p1', split='train', left_seq_id='chainA', right_seq_id='chainB',
                   left_sequence_sha1='a'*40, right_sequence_sha1='b'*40,
                   left_length=100, right_length=200, left_positive=1, right_positive=2,
                   left_observed=100, right_observed=200)
        with path.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            writer.writerow(dict(row, pair_id='p2', left_seq_id='otherA', right_seq_id='otherB'))
        rows, _ = AUDIT.read_manifest(path, 'train', 'sha1')
        assert rows[:, -1].tolist() == [0, 0]


def test_ratio_context_average_differs_only_by_constant():
    p = torch.tensor(.3, requires_grad=True)
    ratios = torch.tensor([.01, .5])
    hard = ((p-ratios)**2).mean()
    soft = (p-ratios.mean())**2
    torch.testing.assert_close(hard-soft, ratios.var(unbiased=False))
    torch.testing.assert_close(torch.autograd.grad(hard, p, retain_graph=True)[0],
                               torch.autograd.grad(soft, p)[0])


def test_marginal_probability_order_need_not_maximize_instance_ap():
    from sklearn.metrics import average_precision_score
    first, second = np.array([1, 1, 0]), np.array([0, 0, 1])
    marginal_order = np.array([.61, .60, .4])
    alternative_order = np.array([.6, .5, .7])
    marginal_ap = .6*average_precision_score(first, marginal_order)+.4*average_precision_score(second, marginal_order)
    alternative_ap = .6*average_precision_score(first, alternative_order)+.4*average_precision_score(second, alternative_order)
    assert marginal_ap == pytest.approx(11/15)
    assert alternative_ap == pytest.approx(.75)
