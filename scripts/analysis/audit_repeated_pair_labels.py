"""Audit validation label diversity for identical ordered sequence inputs, CPU only."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--label-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'val']
    if not rows or len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Empty validation or duplicate pair IDs')
    if args.output.exists():
        raise FileExistsError(args.output)
    rows.sort(key=lambda r: (r['label_path'], int(r['label_offset'])))
    groups, files = {}, {}
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
                if mask not in group['examples']:
                    group['examples'][mask] = {k: row[k] for k in
                        ('pair_id', 'pdb_id', 'assembly_id', 'left_chain_instance_id', 'right_chain_instance_id')}
                    group['examples'][mask].update(side=side, positive_count=len(positive))
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
    summary = {'split': 'val', 'test_read': False, 'complete': True, 'pairs': len(rows),
        'directional_records': 2*len(rows), 'ordered_sequence_groups': len(table),
        'repeated_groups': len(repeated), 'varying_label_groups': len(varying),
        'records_in_varying_groups': sum(r['directional_records'] for r in varying),
        'distinct_input_label_combinations': sum(r['distinct_label_patterns'] for r in table),
        'val_rows_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        'scope': 'Identical ordered sequence UIDs with fully observed residues; label diversity is biological context dependence, not evidence of erroneous labels. Homomer sides share a key. Residue patterns are partner-specific, not unioned.',
        'examples': examples}
    if sum(r['directional_records'] for r in table) != 2*len(rows):
        raise ValueError('Coverage mismatch')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='examples'}), flush=True)


if __name__ == '__main__':
    main()
