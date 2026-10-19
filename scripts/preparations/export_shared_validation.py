"""Export original PDBTM validation labels for an identity-audited intersection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--label-root', type=Path, required=True)
    p.add_argument('--embedding-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-pairs', type=int, default=352)
    a = p.parse_args()
    identity = json.loads(a.identity.read_text())
    hashes, pdbs = set(identity['hashes']), set(identity['pdbs'])
    train_hashes, selected = set(), []
    for r in csv.DictReader(a.manifest.open()):
        if r['split'] == 'train':
            train_hashes.update(r[s + '_sequence_sha1'] for s in ('left', 'right'))
        elif r['split'] == 'val' and r['pdb_id'] in pdbs:
            if all(r[s + '_sequence_sha1'] in hashes for s in ('left', 'right')):
                selected.append(r)
    if train_hashes & hashes:
        raise ValueError('Shared identities overlap old training sequences')
    if not selected:
        raise ValueError('No shared validation pairs')
    if len(selected) != a.expected_pairs or len({r['pair_id'] for r in selected}) != len(selected):
        raise ValueError('Unexpected cohort size or duplicate pair IDs')
    records, cache = [], {}
    for r in selected:
        path = a.label_root / r['label_path']
        d = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('pair_id', 'pdb_id', 'split', 'left_seq_id', 'right_seq_id'):
            if d.get(key) != r[key]:
                raise ValueError(f'{path}: {key} mismatch')
        if float(d.get('cutoff_angstrom', -1)) != 5.5 or d.get('heavy_only') is not True:
            raise ValueError('Unexpected label definition')
        out = dict(r)
        for s in ('left', 'right'):
            sid = r[s + '_seq_id']
            if sid not in cache:
                e = torch.load(a.embedding_root / f'{sid}.pt', map_location='cpu', weights_only=False)
                cache[sid] = e['sequence']
            sequence = cache[sid]
            sha = hashlib.sha1(sequence.encode('ascii')).hexdigest()
            if sha != r[s + '_sequence_sha1'] or sha != d[s + '_sequence_sha1']:
                raise ValueError('Label/sequence hash mismatch')
            raw_y = torch.as_tensor(d[s + '_labels'])
            raw_mask = torch.as_tensor(d[s + '_observed_mask'])
            if not bool(((raw_y == 0) | (raw_y == 1) | (raw_y == -100)).all()):
                raise ValueError('Invalid raw label value')
            if not bool(((raw_mask == 0) | (raw_mask == 1)).all()):
                raise ValueError('Invalid raw mask value')
            y = raw_y.to(torch.long)
            mask = raw_mask.to(torch.bool)
            if y.ndim != 1 or len(y) != len(sequence) or len(y) != int(r[s + '_length']):
                raise ValueError('Length mismatch')
            if not bool(((y == 0) | (y == 1) | (y == -100)).all()):
                raise ValueError('Unexpected label value')
            if not torch.equal(y != -100, mask) or int(mask.sum()) != int(r[s + '_observed']):
                raise ValueError('Mask mismatch')
            if int((y == 1).sum()) != int(r[s + '_positive']):
                raise ValueError('Positive count mismatch')
            out[s + '_sequence'] = sequence
            out[s + '_rcsb_uid'] = 'useq_' + hashlib.sha256(sequence.encode('ascii')).hexdigest()
            out[s + '_labels'] = y.tolist()
        records.append(out)
    payload = {'scope': 'Shared validation candidate; no test labels',
               'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
               'identity_sha256': hashlib.sha256(a.identity.read_bytes()).hexdigest(),
               'old_train_exact_overlap': 0, 'n_pairs': len(records),
               'n_pdb': len({r['pdb_id'] for r in records}), 'records': records}
    with a.output.open('x') as f:
        json.dump(payload, f)
    print(json.dumps({k: v for k, v in payload.items() if k != 'records'}))


if __name__ == '__main__':
    main()
