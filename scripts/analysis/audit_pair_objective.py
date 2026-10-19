"""Read-only audit of dynamic-batch coefficients and weighted smoothed CE."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler


def load_trainer_classes(path):
    source = Path(path).read_text()
    selected = [node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef)
                and node.name in {'PairTokenBatchSampler', 'PairResidueLoss'}]
    if len(selected) != 2:
        raise ValueError('Expected exactly the two frozen trainer classes')
    namespace = dict(torch=torch, nn=nn, BatchSampler=BatchSampler,
                     Iterator=Iterator, random=random)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    hashes = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
              for node in selected}
    return namespace['PairTokenBatchSampler'], namespace['PairResidueLoss'], hashes


def read_manifest(path, split, group_key='uid'):
    if group_key not in ('uid', 'sha1'):
        raise ValueError('Unknown grouping key')
    rows, group_ids, pair_ids = [], {}, set()
    digest = hashlib.sha256()
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != split:
                continue
            if row['pair_id'] in pair_ids:
                raise ValueError('Duplicate pair ID')
            if any(not row.get(key, '').strip()
                   for key in ('pair_id', 'left_seq_id', 'right_seq_id')):
                raise ValueError('Missing pair ID or sequence UID')
            pair_ids.add(row['pair_id'])
            lengths = [int(row[s + '_length']) for s in ('left', 'right')]
            positives = [int(row[s + '_positive']) for s in ('left', 'right')]
            if any(length <= 0 or not 0 <= positive <= length
                   or int(row[s + '_observed']) != length
                   for s, length, positive in zip(('left', 'right'), lengths, positives)):
                raise ValueError('Invalid observed-sequence count')
            suffix = 'seq_id' if group_key == 'uid' else 'sequence_sha1'
            ids = [row.get(s + '_' + suffix, '').strip() for s in ('left', 'right')]
            if not all(ids) or (group_key == 'sha1' and any(
                    len(x) != 40 or any(c not in '0123456789abcdef' for c in x) for x in ids)):
                raise ValueError('Missing or invalid exact-sequence grouping key')
            key = tuple(sorted(ids))
            group = group_ids.setdefault(key, len(group_ids))
            rows.append((*lengths, *positives, group))
            digest.update((json.dumps(row, sort_keys=True) + '\n').encode())
    if not rows:
        raise ValueError('Empty selected manifest')
    return np.asarray(rows, dtype=np.int64), digest.hexdigest()


def batch_coefficients(sampler, count):
    sizes = np.zeros(count, dtype=np.int64)
    histogram = Counter()
    batches = 0
    for batch in sampler:
        if not batch or len(set(batch)) != len(batch) or np.any(sizes[batch]):
            raise ValueError('Empty or duplicate sampler batch')
        sizes[batch] = len(batch)
        histogram[len(batch)] += 1
        batches += 1
    if not batches or np.any(sizes == 0):
        raise ValueError('Sampler did not cover every pair exactly once')
    mass = 1.0 / (batches * sizes)
    if not np.isclose(mass.sum(), 1.0):
        raise ValueError('Coefficient mass does not sum to one')
    return mass, sizes, batches, dict(sorted(histogram.items()))


def target_probability(y, weight, smoothing):
    soft_positive = (1 - smoothing) * y + smoothing / 2
    return weight * soft_positive / (1 - soft_positive + weight * soft_positive)


def loss_probes(loss_class):
    output = []
    for weight in (1.0, 6.93964, 10.380668383505242):
        for smoothing in (0.0, 0.05):
            if smoothing == 0:
                output.append(dict(weight=weight, smoothing=smoothing,
                                   negative_ce_optimum=0.0, positive_ce_optimum=1.0))
                continue
            p0, p1 = [target_probability(y, weight, smoothing) for y in (0, 1)]
            labels = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)
            probabilities = torch.tensor([[p0, p0, p0, p1]], dtype=torch.float64)
            logits = torch.stack((torch.zeros_like(probabilities),
                                  torch.logit(probabilities)), dim=-1).requires_grad_()
            criterion = loss_class(weight, smoothing, 0.0)
            value = criterion._one_side(logits, labels)
            gradient = torch.autograd.grad(value, logits)[0]
            if gradient.abs().max().item() > 2e-6:
                raise ValueError('CE stationary-point probe failed')
            entry = dict(weight=weight, smoothing=smoothing, negative_ce_optimum=p0,
                         positive_ce_optimum=p1,
                         max_abs_stationary_gradient=gradient.abs().max().item())
            coupled = []
            for n_positive in (1, 5, 10, 20, 50):
                labels = (torch.arange(100) < n_positive).long().unsqueeze(0)
                parameter = nn.Parameter(torch.logit(torch.tensor([p0, p1], dtype=torch.float64)))
                optimizer = torch.optim.LBFGS([parameter], max_iter=100,
                    tolerance_grad=1e-8, tolerance_change=1e-12, line_search_fn='strong_wolfe')
                criterion = loss_class(weight, smoothing, 0.25)

                def closure():
                    optimizer.zero_grad()
                    positive_logit = parameter[labels]
                    logits = torch.stack((torch.zeros_like(positive_logit), positive_logit), dim=-1)
                    loss = criterion._one_side(logits, labels)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                final_loss = closure()
                probabilities = parameter.detach().sigmoid()
                coupled.append(dict(length=100, positive=n_positive,
                    negative_probability=probabilities[0].item(),
                    positive_probability=probabilities[1].item(),
                    mean_probability=((100 - n_positive)*probabilities[0]
                                      + n_positive*probabilities[1]).item()/100,
                    loss=final_loss.item(), max_abs_gradient=parameter.grad.abs().max().item()))
            entry['label_aware_two_parameter_toy_with_ratio025'] = coupled
            output.append(entry)
    return output


def summarize_profile(rows, config_snapshot, sampler_class, split):
    config = config_snapshot['config']
    training = config['training']
    sampler = sampler_class(rows[:, :2].tolist(),
        batch_size=training['batch_size' if split == 'train' else 'eval_batch_size'],
        max_tokens=training['max_batch_tokens'], max_pair_cells=training['max_pair_cells'],
        max_single_pair_cells=training['max_single_pair_cells'], shuffle=False,
        seed=training['seed'], bucket_size=training.get('length_bucket_size', 512),
        joint_grid=config['pair_model']['mode'] == 'pairformer')
    mass, sizes, batch_count, histogram = batch_coefficients(sampler, len(rows))
    expected_count = config_snapshot['n_train_pairs' if split == 'train' else 'n_val_pairs']
    if len(rows) != expected_count:
        raise ValueError('Manifest count differs from run config')
    if split == 'train' and batch_count != config_snapshot['train_batches_per_epoch']:
        raise ValueError('Frozen sampler batch count differs from training config')
    train_positive = int(config_snapshot['train_positive'])
    train_negative = int(config_snapshot['train_negative'])
    expected_weight = min(float(training.get('max_pos_weight', 20.0)),
                          train_negative / max(1, train_positive))
    if (train_positive <= 0 or train_negative < 0
            or not np.isclose(config_snapshot['pos_weight'], expected_weight, rtol=1e-10)):
        raise ValueError('Training class counts and pos_weight are inconsistent')
    if split == 'train':
        observed_positive = int(rows[:, 2:4].sum())
        observed_negative = int(rows[:, :2].sum()) - observed_positive
        if (observed_positive, observed_negative) != (train_positive, train_negative):
            raise ValueError('Manifest class counts differ from training config')
    group_counts = np.bincount(rows[:, 4])
    multiplicity = group_counts[rows[:, 4]]
    length, positive = rows[:, :2], rows[:, 2:4]
    prevalence = positive / length
    weight = float(config_snapshot['pos_weight'])
    smoothing = float(training['label_smoothing'])
    denominator = length + (weight - 1) * positive
    positive_coef = weight*((1 - smoothing)*positive + smoothing*length/2)/denominator
    negative_coef = ((1 - smoothing)*(length - positive) + smoothing*length/2)/denominator
    direction_mass = np.broadcast_to(mass[:, None]/2, length.shape)
    groups = {}

    def add(name, mask):
        selected = np.broadcast_to(mask, length.shape)
        coefficient = float(direction_mass[selected].sum())
        groups[name] = dict(directional_records=int(selected.sum()),
            record_fraction=float(selected.mean()), loss_outer_coefficient_fraction=coefficient,
            mean_batch_size=float(np.broadcast_to(sizes[:, None], length.shape)[selected].mean())
                if selected.any() else None,
            mean_prevalence=float(prevalence[selected].mean()) if selected.any() else None)

    for name, mask in [('1', multiplicity == 1), ('2-5', (multiplicity >= 2) & (multiplicity <= 5)),
                       ('6-20', (multiplicity >= 6) & (multiplicity <= 20)),
                       ('>20', multiplicity > 20)]:
        add('multiplicity/' + name, mask[:, None])
    for name, mask in [('<=0.05', prevalence <= .05),
                       ('0.05-0.10', (prevalence > .05) & (prevalence <= .1)),
                       ('0.10-0.20', (prevalence > .1) & (prevalence <= .2)),
                       ('>0.20', prevalence > .2)]:
        add('prevalence/' + name, mask)
    for name, mask in [('<300', length < 300), ('300-599', (length >= 300) & (length < 600)),
                       ('600-999', (length >= 600) & (length < 1000)), ('>=1000', length >= 1000)]:
        add('length/' + name, mask)
    a = float(np.sum(direction_mass * positive_coef))
    b = float(np.sum(direction_mass * negative_coef))
    constant_loss = float(np.sum(direction_mass * (
        np.log(2)*(positive_coef + negative_coef)
        + training['ratio_loss_weight']*(.5 - prevalence)**2)))
    group_mass = np.bincount(rows[:, 4], weights=mass)
    return dict(pairs=len(rows), sequence_pair_groups=len(group_counts),
        batches=batch_count, batch_size_histogram=histogram,
        median_batch_size=float(np.median(np.repeat(list(histogram), list(histogram.values())))),
        median_pair_attended_batch_size=float(np.median(sizes)),
        min_batch_size=int(sizes.min()), max_batch_size=int(sizes.max()),
        min_pair_outer_coefficient=float(mass.min()), max_pair_outer_coefficient=float(mass.max()),
        pair_coefficient_ratio_max_min=float(mass.max()/mass.min()),
        coefficient_only_effective_pairs=float(1/np.sum(mass**2)),
        coefficient_only_effective_groups=float(1/np.sum(group_mass**2)),
        mean_prevalence=float(prevalence.mean()),
        batch_weighted_prevalence=float(np.sum(direction_mass * prevalence)),
        constant_probability_ce_optimum_without_ratio=a/(a+b),
        constant_probability_0p5_actual_loss=constant_loss, groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--trainer', type=Path, required=True)
    parser.add_argument('--config', nargs=2, action='append', required=True, metavar=('NAME', 'JSON'))
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--group-key', choices=('uid', 'sha1'), default='uid')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len({name for name, _ in args.config}) != len(args.config):
        raise ValueError('Duplicate config name')
    torch.set_num_threads(1)
    sampler, loss, class_hashes = load_trainer_classes(args.trainer)
    rows, digest = read_manifest(args.manifest, args.split, args.group_key)
    result = dict(split=args.split, test_labels_read=False, complete=True,
        selected_rows_sha256=digest, trainer_class_ast_sha256=class_hashes,
        trainer_file_sha256=hashlib.sha256(args.trainer.read_bytes()).hexdigest(),
        torch_version=torch.__version__, label_jsonl_read=False, profiles={},
        exact_sequence_group_key=args.group_key, config_sha256={},
        caution='Fixed-parameter loss coefficients, NOT measured gradient shares, Adam update shares, '
                'or independent statistical sample sizes. Ratio toys know the labels and are not model predictions.')
    for name, path in args.config:
        snapshot = json.loads(Path(path).read_text())
        result['config_sha256'][name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        result['profiles'][name] = summarize_profile(rows, snapshot, sampler, args.split)
        print(json.dumps({'completed_profile': name, 'split': args.split}), flush=True)
    result['loss_probes'] = loss_probes(loss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
