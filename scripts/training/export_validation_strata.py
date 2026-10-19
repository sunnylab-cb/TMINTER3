"""Export complete validation-only directional metrics for stratified diagnosis."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_esm_pair import _build_loader, _move_side
from ppcbind.models.esm_pair import ESMPairSiteClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = args.checkpoint.read_bytes()
    import io
    ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    digest = hashlib.sha256(raw).hexdigest()
    del raw
    if ckpt['epoch'] != args.epoch:
        raise ValueError('Checkpoint epoch differs from requested frozen epoch')
    config = ckpt['config']['config']
    if any(config['training'].get(k) is not None for k in
           ('smoke_max_chain_length', 'max_val_pairs')):
        raise ValueError('Refuse a smoke-limited configuration')
    dataset, loader = _build_loader(config, 'val', False)
    rows = {r['pair_id']: r for r in dataset.rows}
    if len(rows) != len(dataset.rows):
        raise ValueError('Duplicate pair IDs')
    train_pdbs, train_sequences = set(), set()
    with Path(config['data']['pair_manifest']).open(newline='') as f:
        for r in csv.DictReader(f):
            if r['split'].lower() == 'train':
                train_pdbs.add(r['pdb_id'].lower())
                train_sequences.update(r[s + '_sequence_sha1'].lower() for s in ('left', 'right'))
    if not train_pdbs or not train_sequences:
        raise ValueError('Training identity audit unavailable')
    val_pdbs = {r['pdb_id'].lower() for r in dataset.rows}
    val_sequences = {r[s + '_sequence_sha1'].lower() for r in dataset.rows for s in ('left', 'right')}
    overlap = {'pdb': len(train_pdbs & val_pdbs), 'exact_sequence': len(train_sequences & val_sequences)}
    if any(overlap.values()):
        raise ValueError('Train/validation identity overlap: ' + str(overlap))
    def pair_key(r):
        return '|'.join(sorted([r['left_sequence_sha1'].lower(),
                                r['right_sequence_sha1'].lower()]))
    counts = Counter(pair_key(r) for r in dataset.rows)
    model = ESMPairSiteClassifier(config['model'], config.get('pair_model', {})).cuda().eval()
    model.load_state_dict(ckpt['model'], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ['pair_id', 'side', 'pdb_id', 'sequence_sha1', 'partner_sha1',
              'sequence_pair', 'pair_multiplicity', 'identical_sequence_pair',
              'sequence_length', 'partner_length', 'n', 'positive_count',
              'positive_fraction', 'ap', 'auroc', 'ce', 'brier', 'mean_probability',
              'tp_0p5', 'fp_0p5', 'fn_0p5', 'tn_0p5']
    seen = set()
    started = time.monotonic()
    with (args.output / 'directional_metrics.csv').open('w') as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, batch in enumerate(loader):
            left, yl = _move_side(batch['left'], torch.device('cuda:0'))
            right, yr = _move_side(batch['right'], torch.device('cuda:0'))
            with torch.amp.autocast('cuda', enabled=bool(config['training'].get('amp', True))):
                out = model(left, right)
            for side, target in [('left', yl), ('right', yr)]:
                logits = out[side + '_logits'].float()
                probs = logits.softmax(-1)[..., 1].cpu().numpy()
                losses = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), target, reduction='none', ignore_index=-100).cpu().numpy()
                labels = target.cpu().numpy()
                other = 'right' if side == 'left' else 'left'
                for idx, pid in enumerate(batch['pair_id']):
                    key = (pid, side)
                    if key in seen:
                        raise ValueError('Duplicate evaluation record')
                    seen.add(key)
                    r = rows[pid]
                    mask = labels[idx] != -100
                    y, p = labels[idx][mask], probs[idx][mask]
                    if not y.size or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
                        raise ValueError('Invalid observed labels or scores')
                    pred = p >= .5
                    writer.writerow(dict(zip(fields, [pid, side, r['pdb_id'],
                        r[side + '_sequence_sha1'].lower(), r[other + '_sequence_sha1'].lower(),
                        pair_key(r), counts[pair_key(r)],
                        int(r['left_sequence_sha1'].lower() == r['right_sequence_sha1'].lower()),
                        int(r[side + '_length']), int(r[other + '_length']), y.size, int(y.sum()),
                        float(y.mean()), float(average_precision_score(y, p)) if y.any() else 0.,
                        float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else '',
                        float(losses[idx][mask].mean()), float(np.mean((p-y)**2)), float(p.mean()),
                        int((pred & (y==1)).sum()), int((pred & (y==0)).sum()),
                        int((~pred & (y==1)).sum()), int((~pred & (y==0)).sum())])))
            if step % 100 == 0:
                handle.flush()
                print(json.dumps({'batches': step+1, 'directional_records': len(seen),
                                  'seconds': time.monotonic()-started}), flush=True)
    if seen != {(pid, s) for pid in rows for s in ('left', 'right')}:
        raise ValueError('Incomplete validation coverage')
    summary = {'checkpoint_sha256': digest, 'epoch': args.epoch, 'split': 'val',
               'test_read': False, 'pairs': len(rows), 'directional_records': len(seen),
               'unique_sequence_pairs': len(counts), 'complete': True,
               'train_val_identity_overlap': overlap,
               'seconds': time.monotonic()-started,
               'config': config,
               'note': 'Native validation, no threshold search. AP=0 for no-positive chains; AUROC undefined for single-class chains.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
