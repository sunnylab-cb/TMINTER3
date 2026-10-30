"""Package unrounded X-PAIR scores in the same directional schema as Ours."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_ours import INPUT_SHA, LABEL_SHA
from evaluate_pilot import CKPTS, MANIFEST_SHA


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'labels', 'run', 'ours', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(digest(args.inputs) == INPUT_SHA and digest(args.labels) == LABEL_SHA,
            'Wrong frozen input or labels')
    inp = json.loads(args.inputs.read_text())
    labs = json.loads(args.labels.read_text())
    run = json.loads((args.run / 'summary.json').read_text())
    ours = json.loads((args.ours / 'summary.json').read_text())
    require(run['status'] == 'complete' and run['input_sha256'] == INPUT_SHA
            and run['manifest_sha256'] == MANIFEST_SHA and not run['smoke']
            and not run['test_evaluation'] and not run['interaction_gating']
            and run['labels_supplied'] is False and run['precision'] == 'float32_unrounded',
            'Wrong source inference run')
    require(ours['input_sha256'] == INPUT_SHA and ours['labels_sha256'] == LABEL_SHA
            and ours['threshold_search'] is False, 'Wrong Ours result')
    by_label = {r['pair_id']: r for r in labs['records']}
    records, ys = [], []
    for ctx in inp['contexts']:
        for side in ('left', 'right'):
            lab = by_label[ctx['pair_id']][side]
            y = np.zeros(lab['length'], dtype=np.int8)
            y[lab['positive']] = 1
            observed = np.sort(lab['observed'])
            ys.append(y[observed])
            records.append({'pair_id': ctx['pair_id'], 'side': side, 'n': len(observed)})
    labels, lengths = np.concatenate(ys), np.array([len(y) for y in ys])
    require(len(records) == 1024 and len(labels) == 289902 and ours['records'] == records,
            'Cohort order or coverage mismatch')
    ours_path = args.ours / 'predictions.npz'
    with np.load(ours_path, allow_pickle=False) as pred:
        require(np.array_equal(pred['labels'], labels) and np.array_equal(pred['lengths'], lengths)
                and pred['probabilities'].shape == labels.shape
                and np.isfinite(pred['probabilities']).all(), 'Ours residue mapping mismatch')
    paths = {name: args.output / name / 'predictions.npz' for name in CKPTS}
    receipt_path = args.output / 'residue_bundle.json'
    require(not receipt_path.exists() and all(not p.exists() for p in paths.values()),
            'Refuse to overwrite a prediction bundle')
    files = {'ours_rcsb8m': {'path': 'ours_rcsb8m/predictions.npz',
                            'sha256': digest(ours_path), 'bytes': ours_path.stat().st_size}}
    require(ours_path.resolve() == (args.output / files['ours_rcsb8m']['path']).resolve(),
            'Ours file must be under the same result root')
    for name, checkpoint_sha in CKPTS.items():
        require(run['models'][name]['completed'] == 428
                and run['models'][name]['checkpoint_sha256'] == checkpoint_sha,
                'Wrong checkpoint or incomplete coverage')
        expected_keys = {ctx['input_key'] for ctx in inp['contexts']}
        require({p.stem for p in (args.run / name).glob('*.npz')} == expected_keys,
                'Wrong source prediction coverage')
        cache, ps = {}, []
        for ctx in inp['contexts']:
            key = ctx['input_key']
            if key not in cache:
                with np.load(args.run / name / (key + '.npz'), allow_pickle=False) as d:
                    require(str(d['checkpoint_sha256']) == checkpoint_sha
                            and str(d['input_sha256']) == INPUT_SHA
                            and str(d['ankh_revision']) == run['ankh_revision']
                            and str(d['ankh_weight_sha256']) == run['ankh_weight_sha256'],
                            'Raw prediction provenance mismatch')
                    cache[key] = {s: (str(d[s + '_uid']), d[s].copy()) for s in ('left', 'right')}
            for side in ('left', 'right'):
                lab = by_label[ctx['pair_id']][side]
                uid, p = cache[key][side]
                require(uid == lab['uid'] == ctx[side + '_uid'] and p.shape == (lab['length'],)
                        and p.dtype == np.float32 and np.isfinite(p).all()
                        and ((p >= 0) & (p <= 1)).all(), 'Invalid residue score mapping')
                ps.append(p[np.sort(lab['observed'])])
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(paths[name], labels=labels, probabilities=np.concatenate(ps), lengths=lengths)
        files[name] = {'path': str(paths[name].relative_to(args.output)),
                       'sha256': digest(paths[name]), 'bytes': paths[name].stat().st_size,
                       'checkpoint_sha256': checkpoint_sha}
    receipt = {'scope': 'same_frozen_val512_not_test', 'input_sha256': INPUT_SHA,
               'labels_sha256': LABEL_SHA, 'directions': 1024, 'residues': len(labels),
               'schema': {'labels': 'flat binary targets', 'probabilities': 'flat unrounded float32 scores',
                          'lengths': '1024 segment lengths; contexts in val512_inputs order, left then right'},
               'files': files}
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
