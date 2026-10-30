"""Export validation labels separately, then score completed label-free inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '682a1815212000991f0ceb9ce2c784aa9085bebf8d0d41fa0e5436836eca4108'
INPUT_SHA = 'ee61e1cb3a7f7fa1213c4f8289cc6fa2a027a260f3778c7509b81af34d7897e2'
CONTRACT = 'observed-polymer-heavy-atom-5p5-assembly-v2'
CKPTS = {'interface_pioneer': '7ab260dc7917205367cb56ba8f706b36ffa502540c44a5395a1a51457a4c6bd1',
         'multitask_xfair': 'd8c50d03cd6107d2b411d7095a34243ec1b31e6b7cc1d590329c9328c1e8df72'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(a):
    require(digest(a.manifest) == MANIFEST_SHA, 'Wrong validation manifest')
    rows = list(csv.DictReader(a.manifest.open()))
    require(len(rows) == 512 and all(r['split'] == 'val' for r in rows), 'Invalid cohort')
    records = []
    for row in rows:
        rel = Path(row['label_path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe label path')
        with (a.labels_root / rel).open('rb') as f:
            f.seek(int(row['label_offset']))
            raw = f.readline()
        require(raw.endswith(b'\n'), 'Wrong label offset')
        d = json.loads(raw)
        require(d['contract'] == CONTRACT, 'Wrong contact definition')
        for k in ('pair_id', 'pdb_id', 'assembly_id', 'split', 'left_seq_id', 'right_seq_id'):
            require(str(d[k]) == row[k], 'Label metadata mismatch: ' + k)
        contacts = d['residue_contacts']
        lengths = [int(row[s + '_length']) for s in ('left', 'right')]
        require(isinstance(contacts, list) and len(contacts) == int(row['n_contacts']) > 0,
                'Invalid contact count')
        require(all(isinstance(c, list) and len(c) == 2 and
                    all(type(i) is int and 0 <= i < length for i, length in zip(c, lengths))
                    for c in contacts), 'Invalid contact index')
        require(len({tuple(c) for c in contacts}) == len(contacts), 'Duplicate contacts')
        rec = {'pair_id': row['pair_id'], 'pdb_id': row['pdb_id'],
               'label_record_sha256': hashlib.sha256(raw).hexdigest()}
        for col, side in enumerate(('left', 'right')):
            length = lengths[col]
            pos = d[side + '_positive']
            require(isinstance(pos, list) and all(type(i) is int for i in pos), 'Invalid positive indices')
            require(len(pos) == len(set(pos)) == int(row[side + '_positive'])
                    and set(pos) == {c[col] for c in contacts}, 'Wrong positive projection')
            observed = (list(range(length)) if d.get('observed_sequence_only') is True
                        and int(row[side + '_observed']) == length else d[side + '_observed_indices'])
            require(all(type(i) is int and 0 <= i < length for i in observed)
                    and len(observed) == len(set(observed)) == int(row[side + '_observed'])
                    and set(pos) <= set(observed), 'Invalid observed mask')
            rec[side] = {'uid': row[side + '_seq_id'], 'length': length,
                         'positive': pos, 'observed': observed}
        records.append(rec)
    require(len({r['pair_id'] for r in records}) == 512, 'Duplicate contexts')
    require(not a.output.exists(), 'Refuse to overwrite frozen labels')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'scope': 'validation_only_evaluation_labels',
                                   'contract': CONTRACT, 'manifest_sha256': MANIFEST_SHA,
                                   'geometry_recomputed': False,
                                   'label_trust': 'existing_training_label_pipeline; projection_QC_only',
                                   'records': records}) + '\n')
    print(json.dumps({'records': len(records), 'sha256': digest(a.output)}))


def score(a):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
    require(digest(a.inputs) == INPUT_SHA, 'Wrong frozen input')
    require(digest(a.labels) == a.labels_sha256, 'Evaluation label snapshot changed')
    inp, labels = json.loads(a.inputs.read_text()), json.loads(a.labels.read_text())
    run = json.loads((a.run / 'summary.json').read_text())
    require(run['status'] == 'complete' and run['smoke'] is False and run['input_pairs'] == 428
            and run['input_sha256'] == INPUT_SHA and run['manifest_sha256'] == MANIFEST_SHA
            and run['labels_supplied'] is False and run['interaction_gating'] is False
            and run['test_evaluation'] is False
            and run['precision'] == 'float32_unrounded',
            'Incomplete or wrong inference run')
    require(labels['scope'] == 'validation_only_evaluation_labels' and labels['contract'] == CONTRACT
            and labels['manifest_sha256'] == MANIFEST_SHA, 'Wrong evaluation label contract')
    contexts = {r['pair_id']: r for r in inp['contexts']}
    expected_keys = {r['input_key'] for r in contexts.values()}
    require(len(expected_keys) == 428, 'Distinct ordered input coverage changed')
    require(len(labels['records']) == 512 and {r['pair_id'] for r in labels['records']} == set(contexts),
            'Label/inference cohort mismatch')
    a.output.mkdir(parents=True, exist_ok=False)

    def classification(y, p, gate):
        call = p > gate
        tp = int(np.sum(call & (y == 1))); fp = int(np.sum(call & (y == 0)))
        fn = int(np.sum(~call & (y == 1))); tn = int(np.sum(~call & (y == 0)))
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return {'gate': float(gate), 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                'mcc': (tp * tn - fp * fn) / den if den else 0.0}

    report = {'scope': 'frozen_val512_diagnostic_not_test', 'test_evaluation': False,
              'manifest_sha256': MANIFEST_SHA, 'input_sha256': INPUT_SHA,
              'label_export_sha256': digest(a.labels), 'contract': CONTRACT,
              'geometry_recomputed': False,
              'label_trust': 'existing_training_label_pipeline; no new atom-distance audit',
              'threshold_operator': '>', 'val_gate_grid': '0.01..0.99 step 0.01; lowest gate wins ties',
              'weighting': 'equal original directional contexts for macro AP; pooled residues for pooled metrics',
              'models': {}}
    for name, checkpoint_sha in CKPTS.items():
        require(run['models'][name]['completed'] == 428
                and run['models'][name]['checkpoint_sha256'] == checkpoint_sha, 'Wrong checkpoint coverage')
        require({p.stem for p in (a.run / name).glob('*.npz')} == expected_keys,
                'Prediction file coverage mismatch')
        scores, per, ys, ps = {}, [], [], []
        for rec in labels['records']:
            ctx = contexts[rec['pair_id']]
            require(rec['pdb_id'] == ctx['pdb_id'], 'Context PDB mismatch')
            key = ctx['input_key']
            if key not in scores:
                with np.load(a.run / name / (key + '.npz'), allow_pickle=False) as d:
                    require(str(d['checkpoint_sha256']) == checkpoint_sha
                            and str(d['input_sha256']) == INPUT_SHA
                            and str(d['ankh_revision']) == run['ankh_revision']
                            and str(d['ankh_weight_sha256']) == run['ankh_weight_sha256'],
                            'Prediction provenance mismatch')
                    scores[key] = {s: (str(d[s + '_uid']), d[s].copy()) for s in ('left', 'right')}
            for side in ('left', 'right'):
                lab = rec[side]; uid, pred = scores[key][side]
                require(pred.ndim == 1 and pred.dtype == np.float32
                        and uid == lab['uid'] == ctx[side + '_uid'] and len(pred) == lab['length']
                        == len(inp['sequences'][uid]) and np.isfinite(pred).all()
                        and (pred >= 0).all() and (pred <= 1).all(), 'Residue alignment/score mismatch')
                require(len(lab['observed']) == len(set(lab['observed'])) and
                        all(type(i) is int and 0 <= i < len(pred) for i in lab['observed']), 'Invalid mask')
                require(all(type(i) is int and i in lab['observed'] for i in lab['positive'])
                        and len(lab['positive']) == len(set(lab['positive'])), 'Invalid labels')
                y = np.zeros(len(pred), dtype=np.int8); y[lab['positive']] = 1
                mask = np.array(lab['observed'], dtype=int); y, pred = y[mask], pred[mask]
                require(len(y) > 0 and int(y.sum()) > 0, 'Undefined directional AP')
                ap = float(average_precision_score(y, pred))
                roc = float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else None
                per.append({'pair_id': rec['pair_id'], 'pdb_id': rec['pdb_id'], 'side': side,
                            'uid': uid, 'length': len(y), 'positive': int(y.sum()), 'ap': ap,
                            'auroc': roc, **classification(y, pred, 0.5)})
                ys.append(y); ps.append(pred)
        require(len(per) == 1024, 'Directional context count mismatch')
        require(set(scores) == expected_keys, 'Not all ordered inputs were loaded')
        y, pred = np.concatenate(ys), np.concatenate(ps)
        precision, recall, _ = precision_recall_curve(y, pred)
        gates = [classification(y, pred, i / 100) for i in range(1, 100)]
        result = {'coverage_pair_contexts': 512, 'directional_contexts': len(per),
                  'residues': len(y), 'positive_fraction': float(y.mean()),
                  'macro_ap': float(np.mean([r['ap'] for r in per])),
                  'pooled_ap': float(average_precision_score(y, pred)),
                  'pooled_pr_auc_trapezoid': float(auc(recall, precision)),
                  'pooled_auroc': float(roc_auc_score(y, pred)),
                  'official_0p5': classification(y, pred, 0.5),
                  'val_best_f1': max(gates, key=lambda r: r['f1']),
                  'val_best_mcc': max(gates, key=lambda r: r['mcc'])}
        report['models'][name] = result
        with (a.output / (name + '_per_direction.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(per[0])); writer.writeheader(); writer.writerows(per)
    (a.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export'); e.add_argument('--manifest', type=Path, required=True)
    e.add_argument('--labels-root', type=Path, required=True); e.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('score')
    for flag in ('labels', 'inputs', 'run', 'output'):
        s.add_argument('--' + flag, type=Path, required=True)
    s.add_argument('--labels-sha256', required=True,
                   help='Frozen export digest from the trusted label export receipt, not a model-selected artifact')
    a = p.parse_args()
    (export if a.command == 'export' else score)(a)


if __name__ == '__main__':
    main()
