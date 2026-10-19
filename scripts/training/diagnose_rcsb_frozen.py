"""Validation-only inference interventions; not retrained ablations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from train_esm_pair import _build_loader, _move_side
from ppcbind.data.esm_pair_dataset import collate_esm_pair_features
from ppcbind.models.esm_pair import ESMPairSiteClassifier


def replacement_index(rows, indices, query, donor_side):
    excluded = {query['left_sequence_sha1'].lower(), query['right_sequence_sha1'].lower()}
    candidates = [i for i in indices if rows[i]['pdb_id'].lower() != query['pdb_id'].lower()
                  and rows[i][donor_side + '_sequence_sha1'].lower() not in excluded]
    if not candidates:
        raise ValueError('No eligible validation replacement partner')
    return min(candidates, key=lambda i: (
        abs(int(rows[i][donor_side + '_length']) - int(query[donor_side + '_length'])),
        rows[i]['pair_id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--replace-partner', action='store_true')
    args = parser.parse_args()
    if args.count <= 0 or args.max_length <= 0:
        parser.error('count and max-length must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    raw = args.checkpoint.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    import io
    ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    del raw
    config = ckpt['config']['config']
    if (config['data'].get('pair_data_backend') != 'rcsb_observed'
            or config['data'].get('rcsb_feature_mode') != 'spatial_scalar89_train_norm_plus_availability90'
            or config['model'].get('d_seq') != 90):
        raise ValueError('Requires the RCSB normalized scalar89 plus availability90 contract')
    if config['pair_model']['mode'] != 'outer_product':
        raise ValueError('Only outer-product checkpoints are supported')
    dataset, _ = _build_loader(config, 'val', False)
    eligible = [i for i, r in enumerate(dataset.rows)
                if max(int(r['left_length']), int(r['right_length'])) <= args.max_length]
    indices = sorted(eligible, key=lambda i: hashlib.sha256(
        ('probe20261018' + dataset.rows[i]['pair_id']).encode()).digest())[:args.count]
    if not indices:
        raise ValueError('Empty diagnostic cohort')
    model = ESMPairSiteClassifier(config['model'], config['pair_model']).cuda().eval()
    model.load_state_dict(ckpt['model'], strict=True)
    amp = bool(config['training'].get('amp', True))
    records = []
    with torch.inference_mode():
        for step, idx in enumerate(indices):
            row = dataset.rows[idx]
            batch = collate_esm_pair_features([dataset[idx]])
            left, yl = _move_side(batch['left'], torch.device('cuda:0'))
            right, yr = _move_side(batch['right'], torch.device('cuda:0'))
            results = {}
            for variant in ('full', 'without_partner_delta', 'zero_ps'):
                model.mode = 'independent' if variant == 'without_partner_delta' else 'outer_product'
                a, b = dict(left), dict(right)
                if variant == 'zero_ps':
                    # Zero normalized descriptors, retain the availability bit.
                    for inputs in (a, b):
                        inputs['seq_features'] = inputs['seq_features'].clone()
                        inputs['seq_features'][..., :89] = 0
                with torch.amp.autocast('cuda', enabled=amp):
                    out = model(a, b)
                results[variant] = out
            replacements = {}
            if args.replace_partner:
                model.mode = 'outer_product'
                replaced = {}
                for query_side, donor_side in (('left', 'right'), ('right', 'left')):
                    donor_idx = replacement_index(dataset.rows, eligible, row, donor_side)
                    donor_row = dataset.rows[donor_idx]
                    donor_batch = collate_esm_pair_features([dataset[donor_idx]])
                    donor, _ = _move_side(donor_batch[donor_side], torch.device('cuda:0'))
                    with torch.amp.autocast('cuda', enabled=amp):
                        out = model(left, donor) if query_side == 'left' else model(donor, right)
                    replaced[query_side + '_logits'] = out[query_side + '_logits']
                    replacements[query_side] = {
                        'replacement_seq_id': donor_row[donor_side + '_seq_id'],
                        'replacement_pdb_id': donor_row['pdb_id'],
                        'original_partner_length': int(row[donor_side + '_length']),
                        'replacement_length': int(donor_row[donor_side + '_length'])}
                results['replaced_partner'] = replaced
            for side, y in (('left', yl), ('right', yr)):
                keep = y[0] != -100
                if not bool(keep.any()):
                    raise ValueError(f"No observed residues: {row['pair_id']}/{side}")
                labels = y[0][keep].cpu().numpy()
                record = {'pair_id': row['pair_id'], 'side': side, 'pdb_id': row['pdb_id'],
                          'homomer': row['left_sequence_sha1'] == row['right_sequence_sha1'],
                          'length': len(labels), 'positive_fraction': float(labels.mean())}
                record.update(replacements.get(side, {}))
                reference = results['full'][side + '_logits'][0][keep].float().softmax(-1)[:, 1]
                for variant, out in results.items():
                    logits = out[side + '_logits'][0][keep].float()
                    prob = logits.softmax(-1)[:, 1]
                    record[variant + '_ap'] = float(average_precision_score(labels, prob.cpu().numpy())) if labels.any() else 0.0
                    record[variant + '_ce'] = float(torch.nn.functional.cross_entropy(logits, y[0][keep]))
                    record[variant + '_prob_change'] = float((prob - reference).abs().mean())
                records.append(record)
            if step % 32 == 0:
                print(json.dumps({'completed_pairs': step + 1, 'total': len(indices)}), flush=True)
    groups = {}
    for name, subset in [('all', records), ('homomer', [r for r in records if r['homomer']]),
                         ('heteromer', [r for r in records if not r['homomer']])]:
        groups[name] = {'n': len(subset)}
        if subset:
            for key in records[0]:
                if key.endswith(('_ap', '_ce', '_prob_change')):
                    groups[name][key] = float(np.mean([r[key] for r in subset]))
    payload = {'checkpoint_sha256': digest, 'epoch': ckpt.get('epoch'),
               'checkpoint_path': str(args.checkpoint),
               'aggregation': 'Equal weight per directional chain-side record, not pooled residues',
               'split': 'val', 'test_read': False, 'eligible_pairs': len(eligible),
               'selected_pairs': len(indices), 'max_length': args.max_length,
               'partner_replacement': args.replace_partner,
               'replacement_policy': 'Nearest-length partner among all length-eligible validation pairs; different PDB and sequence from both original chains; not a biological negative',
               'scope': 'Hash-selected validation subset; interventions are not retrained ablations',
               'groups': groups, 'records': records}
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
