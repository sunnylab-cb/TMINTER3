"""Audit train/validation label diversity for identical ordered inputs, CPU only."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np


def pattern_metrics(patterns, length):
    """Empirical mask diversity, not an AP ceiling or population noise estimate."""
    count = sum(patterns.values())
    positive_counts = np.zeros(length, dtype=np.float64)
    modal = min(patterns, key=lambda mask: (-patterns[mask], mask))
    modal_jaccard = 0.0
    for mask, frequency in patterns.items():
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            positive_counts[bit.bit_length() - 1] += frequency
            remaining ^= bit
        union = (mask | modal).bit_count()
        modal_jaccard += frequency * ((mask & modal).bit_count() / union if union else 1.0)
    q = positive_counts / count
    variable = (q > 0) & (q < 1)
    entropy = np.zeros(length)
    entropy[variable] = -q[variable]*np.log(q[variable]) - (1-q[variable])*np.log1p(-q[variable])
    disagreement = float(np.mean(2*q*(1-q))*count/(count-1)) if count > 1 else 0.0
    return dict(empirical_entropy_nats_per_residue=float(entropy.mean()),
                expected_distinct_record_disagreement=disagreement,
                mean_jaccard_to_modal_mask=modal_jaccard/count,
                positive_count_min=min(mask.bit_count() for mask in patterns),
                positive_count_max=max(mask.bit_count() for mask in patterns))


def summarize_contexts(contexts):
    """Fixed side separates two orientations from distinct pairs in an assembly."""
    levels = {}
    for name, prefix in (('within_assembly_fixed_side', 5), ('within_assembly', 4),
                         ('within_pdb', 3), ('all_pdbs', 2)):
        merged = defaultdict(Counter)
        for key, patterns in contexts.items():
            merged[key[:prefix]].update(patterns)
        repeated = [p for p in merged.values() if sum(p.values()) > 1]
        varying = [p for p in repeated if len(p) > 1]
        levels[name] = dict(groups=len(merged), repeated_groups=len(repeated),
            varying_groups=len(varying), records_in_varying_groups=sum(sum(p.values()) for p in varying))
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--label-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--context-detail', action='store_true')
    args = parser.parse_args()
    with args.manifest.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == args.split]
    if not rows or len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Empty selected split or duplicate pair IDs')
    if args.output.exists():
        raise FileExistsError(args.output)
    rows.sort(key=lambda r: (r['label_path'], int(r['label_offset'])))
    groups, files = {}, {}
    observed_encodings = Counter()
    contexts = defaultdict(Counter)
    homomer_sides = Counter()
    started = time.monotonic()
    with ExitStack() as stack:
        for i, row in enumerate(rows):
            path = Path(row['label_path'])
            if not path.is_absolute():
                path = args.label_root / path
            if path not in files:
                files[path] = stack.enter_context(path.open('rb'))
            handle = files[path]
            handle.seek(int(row['label_offset']))
            raw = handle.readline()
            if not raw.endswith(b'\n'):
                raise ValueError('Incomplete label record')
            label = json.loads(raw)
            for key in ('pair_id', 'pdb_id', 'assembly_id', 'split', 'left_seq_id', 'right_seq_id'):
                if str(label[key]) != row[key]:
                    raise ValueError('Manifest/label mismatch: ' + key)
            if (label.get('contract') != 'observed-polymer-heavy-atom-5p5-assembly-v2'
                    or label.get('observed_sequence_only') is not True
                    or label.get('heavy_only') is not True or label.get('cutoff_angstrom') != 5.5):
                raise ValueError('Unexpected label contract')
            contacts = label.get('residue_contacts')
            if not isinstance(contacts, list) or not contacts:
                raise ValueError('Missing contact pairs')
            projected = [set(), set()]
            seen_contacts = set()
            for contact in contacts:
                if (not isinstance(contact, list) or len(contact) != 2
                        or any(type(j) is not int or not 0 <= j < int(row[s + '_length'])
                               for j, s in zip(contact, ('left', 'right')))):
                    raise ValueError('Invalid contact pair')
                seen_contacts.add(tuple(contact))
                for j in (0, 1):
                    projected[j].add(contact[j])
            if len(seen_contacts) != len(contacts) or len(contacts) != int(row['n_contacts']):
                raise ValueError('Contact count or uniqueness mismatch')
            for side, other in (('left', 'right'), ('right', 'left')):
                length = int(row[side + '_length'])
                if int(row[side + '_observed']) != length:
                    raise ValueError('This audit requires fully observed coordinate sequences')
                observed = label.get(side + '_observed_indices')
                # The frozen materializer canonically stores the all-observed
                # flag and count, not a redundant range(length) vector.
                if side + '_observed_indices' not in label:
                    if label.get('observed_sequence_only') is not True or int(label[side + '_observed']) != length:
                        raise ValueError('Missing explicit observed indices and all-observed contract')
                    observed_encodings['all_observed_contract'] += 1
                elif (not isinstance(observed, list) or len(observed) != length
                        or any(type(j) is not int for j in observed)
                        or set(observed) != set(range(length))):
                    raise ValueError('Label observed indices must cover the coordinate sequence')
                else:
                    observed_encodings['explicit_indices'] += 1
                positive = label[side + '_positive']
                if (not isinstance(positive, list)
                        or len(positive) != int(row[side + '_positive']) or len(set(positive)) != len(positive)
                        or any(type(j) is not int or not 0 <= j < length for j in positive)):
                    raise ValueError('Invalid positive indices')
                if set(positive) != projected[0 if side == 'left' else 1]:
                    raise ValueError('Positive labels differ from contact projection')
                mask = sum(1 << j for j in positive)
                key = (row[side + '_seq_id'], row[other + '_seq_id'])
                group = groups.setdefault(key, {'length': length, 'n': 0, 'pdbs': set(),
                                                'patterns': Counter(), 'examples': {}})
                if length != group['length']:
                    raise ValueError('Identical sequence UID with inconsistent length')
                group['n'] += 1
                group['pdbs'].add(row['pdb_id'])
                group['patterns'][mask] += 1
                if args.context_detail:
                    contexts[(*key, row['pdb_id'], row['assembly_id'], side)][mask] += 1
                if mask not in group['examples']:
                    group['examples'][mask] = {k: row[k] for k in
                        ('pair_id', 'pdb_id', 'assembly_id', 'left_chain_instance_id', 'right_chain_instance_id')}
                    group['examples'][mask].update(side=side, positive_count=len(positive))
            if args.context_detail and row['left_seq_id'] == row['right_seq_id']:
                a, b = projected
                homomer_sides['pairs'] += 1
                homomer_sides['different_masks'] += int(a != b)
                homomer_sides['jaccard_sum'] += len(a & b) / len(a | b)
                homomer_sides['disagreement_sum'] += len(a ^ b) / int(row['left_length'])
            if (i+1) % 10000 == 0:
                print(json.dumps({'pairs': i+1, 'total': len(rows), 'seconds': time.monotonic()-started}), flush=True)

    fields = ['query_uid', 'partner_uid', 'length', 'directional_records', 'pdbs',
              'distinct_label_patterns', 'largest_identical_label_count',
              'variable_residue_positions', 'union_positive', 'intersection_positive']
    table = []
    for (query, partner), group in groups.items():
        masks = list(group['patterns'])
        union, intersection = 0, (1 << group['length']) - 1
        for mask in masks:
            union |= mask
            intersection &= mask
        table.append(dict(zip(fields, [query, partner, group['length'], group['n'],
            len(group['pdbs']), len(masks), max(group['patterns'].values()),
            (union ^ intersection).bit_count(), union.bit_count(), intersection.bit_count()])))
        if args.context_detail:
            table[-1].update(pattern_metrics(group['patterns'], group['length']))
    if args.context_detail:
        fields = list(table[0])
    table.sort(key=lambda r: (-r['directional_records'], r['query_uid'], r['partner_uid']))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'ordered_sequence_groups.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    examples = []
    for row in table[:20]:
        group = groups[(row['query_uid'], row['partner_uid'])]
        masks = sorted(group['patterns'], key=lambda m: (-group['patterns'][m], m))
        first = masks[0]
        second = min(masks, key=lambda m: ((m & first).bit_count()/max(1, (m | first).bit_count()), m))
        examples.append({**row, 'modal_pattern': group['examples'][first],
            'least_overlap_with_modal': group['examples'][second],
            'jaccard': (first & second).bit_count()/max(1, (first | second).bit_count()),
            'pattern_frequencies': sorted(group['patterns'].values(), reverse=True)})
    repeated = [r for r in table if r['directional_records'] > 1]
    varying = [r for r in repeated if r['distinct_label_patterns'] > 1]
    summary = {'split': args.split, 'test_read': False, 'complete': True, 'pairs': len(rows),
        'directional_records': 2*len(rows), 'ordered_sequence_groups': len(table),
        'repeated_groups': len(repeated), 'varying_label_groups': len(varying),
        'records_in_varying_groups': sum(r['directional_records'] for r in varying),
        'distinct_input_label_combinations': sum(r['distinct_label_patterns'] for r in table),
        'observed_encoding_counts': dict(observed_encodings),
        args.split + '_rows_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        'scope': 'Identical ordered sequence UIDs with fully observed residues; label diversity is compatible with context dependence, not by itself evidence of erroneous labels. Homomer sides share a key. Residue patterns are partner-specific, not unioned.',
        'examples': examples}
    if args.context_detail:
        summary['context_levels'] = summarize_contexts(contexts)
        summary['context_levels_caution'] = 'Nested overlapping counts, not a mutually exclusive causal decomposition. Fixed side keeps left and right orientations separate.'
        summary['homomer_same_pair_sides'] = dict(homomer_sides)
        if homomer_sides['pairs']:
            summary['homomer_same_pair_sides'].update(
                mean_jaccard=homomer_sides['jaccard_sum']/homomer_sides['pairs'],
                mean_disagreement=homomer_sides['disagreement_sum']/homomer_sides['pairs'])
        summary['mask_effect_size'] = {}
        for name, subset in [('all_groups', table), ('repeated_groups', repeated), ('varying_groups', varying)]:
            stats = dict(groups=len(subset))
            for metric in ('empirical_entropy_nats_per_residue', 'expected_distinct_record_disagreement',
                           'mean_jaccard_to_modal_mask'):
                values = np.asarray([r[metric] for r in subset])
                weights = np.asarray([r['directional_records'] for r in subset])
                stats[metric] = dict(group_macro_mean=float(values.mean()),
                    group_median=float(np.median(values)),
                    record_weighted_mean=float(np.average(values, weights=weights))) if len(values) else None
            summary['mask_effect_size'][name] = stats
    if sum(r['directional_records'] for r in table) != 2*len(rows):
        raise ValueError('Coverage mismatch')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='examples'}), flush=True)


if __name__ == '__main__':
    main()
