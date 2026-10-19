"""Same-label, shared-validation evaluation for legacy and RCSB checkpoints."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_esm_pair import _build_loader, _move_side
from ppcbind.data.esm_pair_dataset import collate_esm_pair_features
from ppcbind.models.esm_pair import ESMPairSiteClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--expected-pairs', type=int, default=351)
    a = p.parse_args()
    raw = a.checkpoint.read_bytes()
    ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    digest = hashlib.sha256(raw).hexdigest()
    del raw
    if ckpt['epoch'] != a.epoch:
        raise ValueError(f"Expected epoch {a.epoch}, got {ckpt['epoch']}")
    config = ckpt['config']['config']
    cohort_raw = a.cohort.read_bytes()
    cohort = json.loads(cohort_raw)['records']
    if a.expected_pairs <= 0 or len(cohort) != a.expected_pairs or len({r['pair_id'] for r in cohort}) != a.expected_pairs:
        raise ValueError('Unexpected shared cohort')
    hashes = {r[s + '_sequence_sha1'] for r in cohort for s in ('left', 'right')}
    val_hashes, train_hashes, val_pdbs, train_pdbs = set(), set(), set(), set()
    val_pairs = set()
    for r in csv.DictReader(open(config['data']['pair_manifest'])):
        if r['split'] not in ('train', 'val'):
            continue
        hs = train_hashes if r['split'] == 'train' else val_hashes
        ps = train_pdbs if r['split'] == 'train' else val_pdbs
        hs.update(r[s + '_sequence_sha1'] for s in ('left', 'right'))
        ps.add(r['pdb_id'])
        if r['split'] == 'val':
            val_pairs.add((r['pdb_id'], *sorted((r['left_sequence_sha1'], r['right_sequence_sha1']))))
    pdbs = {r['pdb_id'] for r in cohort}
    if hashes & train_hashes or pdbs & train_pdbs or not hashes <= val_hashes or not pdbs <= val_pdbs:
        raise ValueError('Shared cohort violates checkpoint train/validation identity separation')
    for r in cohort:
        key = (r['pdb_id'], *sorted((r['left_sequence_sha1'], r['right_sequence_sha1'])))
        if key not in val_pairs:
            raise ValueError(f"Shared sequence pair not found in validation PDB: {r['pair_id']}")
        for side in ('left', 'right'):
            labels = torch.as_tensor(r[side + '_labels'])
            sequence = r[side + '_sequence']
            if labels.ndim != 1 or len(labels) != len(sequence) or len(labels) != int(r[side + '_length']):
                raise ValueError('Invalid shared label length')
            if not bool(((labels == -100) | (labels == 0) | (labels == 1)).all()):
                raise ValueError('Invalid shared label values')
            if int((labels != -100).sum()) != int(r[side + '_observed']) or int((labels == 1).sum()) != int(r[side + '_positive']):
                raise ValueError('Invalid shared label counts')
    torch.set_num_threads(4)
    dataset, _ = _build_loader(config, 'val', False)
    lookup = {r['pair_id']: i for i, r in enumerate(dataset.rows)}
    backend = config['data'].get('pair_data_backend', 'legacy')
    model = ESMPairSiteClassifier(config['model'], config['pair_model']).cuda().eval()
    model.load_state_dict(ckpt['model'], strict=True)
    a.output.mkdir(parents=True, exist_ok=False)
    results, all_y, all_p = [], [], []
    with torch.inference_mode():
        for index, r in enumerate(cohort):
            if backend == 'legacy':
                source = dataset.rows[lookup[r['pair_id']]]
                for key in ('pdb_id', 'left_sequence_sha1', 'right_sequence_sha1'):
                    if source[key] != r[key]:
                        raise ValueError('Legacy pair metadata differs from cohort')
                item = dataset[lookup[r['pair_id']]]
            elif backend == 'rcsb_observed':
                item = {'pair_id': r['pair_id'], 'pdb_id': r['pdb_id'],
                        'contact_index': torch.empty((2, 0), dtype=torch.long)}
                for side in ('left', 'right'):
                    seq = r[side + '_sequence']
                    if hashlib.sha1(seq.encode()).hexdigest() != r[side + '_sequence_sha1']:
                        raise ValueError('Export sequence SHA1 mismatch')
                    if 'useq_' + hashlib.sha256(seq.encode()).hexdigest() != r[side + '_rcsb_uid']:
                        raise ValueError('Export UID mismatch')
                    item[side] = dataset._load_side(r[side + '_rcsb_uid'], r[side + '_sequence_sha1'], r['pair_id'])
                    item[side]['labels'] = torch.tensor(r[side + '_labels'], dtype=torch.long)
                    item[side + '_observed_mask'] = item[side]['labels'] != -100
            else:
                raise ValueError('Unsupported backend')
            for side in ('left', 'right'):
                if not torch.equal(item[side]['labels'], torch.tensor(r[side + '_labels'])):
                    raise ValueError('Labels differ from shared export')
            batch = collate_esm_pair_features([item])
            left, yl = _move_side(batch['left'], torch.device('cuda:0'))
            right, yr = _move_side(batch['right'], torch.device('cuda:0'))
            with torch.amp.autocast('cuda', enabled=config['training'].get('amp', True)):
                out = model(left, right)
            for side, y in (('left', yl), ('right', yr)):
                keep = y[0] != -100
                if not bool(keep.any()):
                    raise ValueError('Empty observed chain')
                logits = out[side + '_logits'][0][keep].float()
                prob = logits.softmax(-1)[:, 1].cpu().numpy()
                labels = y[0][keep].cpu().numpy()
                results.append({'pair_id': r['pair_id'], 'side': side, 'pdb_id': r['pdb_id'],
                                'ap': float(average_precision_score(labels, prob)) if labels.any() else 0.,
                                'ce': float(torch.nn.functional.cross_entropy(logits, y[0][keep])),
                                'positive_fraction': float(labels.mean()), 'n': len(labels)})
                all_y.append(labels)
                all_p.append(prob)
            if index % 64 == 0:
                print(json.dumps({'pairs_completed': index + 1, 'total': len(cohort)}), flush=True)
    y, prob = np.concatenate(all_y), np.concatenate(all_p)
    summary = {'checkpoint_sha256': digest, 'cohort_sha256': hashlib.sha256(cohort_raw).hexdigest(),
               'epoch': a.epoch, 'backend': backend, 'test_read': False,
               'n_pairs': len(cohort), 'n_residues': len(y), 'positive_fraction': float(y.mean()),
               'macro_ap': float(np.mean([r['ap'] for r in results])),
               'macro_ce': float(np.mean([r['ce'] for r in results])),
               'pooled_ap': float(average_precision_score(y, prob)),
               'auroc': float(roc_auc_score(y, prob)), 'records': results}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    np.savez_compressed(a.output / 'predictions.npz', labels=y, probabilities=prob,
                        lengths=np.array([r['n'] for r in results]))
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
