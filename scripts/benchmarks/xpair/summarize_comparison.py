"""Verify Ours on the frozen X-PAIR cohort and make a common-threshold table."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    auc, average_precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score,
)

from compare_ours import CHECKPOINT_SHA, INPUT_SHA, LABEL_SHA, TRAIN_MANIFEST_SHA


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for flag in ('inputs', 'labels', 'ours', 'xpair-metrics', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    require(digest(args.inputs) == INPUT_SHA and digest(args.labels) == LABEL_SHA,
            'Wrong cohort or label snapshot')
    inp = json.loads(args.inputs.read_text())
    labels = json.loads(args.labels.read_text())
    ours = json.loads((args.ours / 'summary.json').read_text())
    xpair = json.loads(args.xpair_metrics.read_text())
    require(ours['checkpoint_sha256'] == CHECKPOINT_SHA and ours['epoch'] == 5
            and ours['input_sha256'] == INPUT_SHA and ours['labels_sha256'] == LABEL_SHA
            and ours['train_manifest_sha256'] == TRAIN_MANIFEST_SHA
            and ours['train_overlap'] == {'exact_sequence': 0, 'pdb': 0}
            and ours['threshold_search'] is False and ours['test_evaluation'] is False,
            'Wrong Ours evaluation provenance')
    require(xpair['input_sha256'] == INPUT_SHA
            and xpair['label_export_sha256'] == LABEL_SHA
            and xpair['threshold_operator'] == '>' and xpair['test_evaluation'] is False,
            'X-PAIR metrics use a different evaluation contract')
    by_label = {r['pair_id']: r for r in labels['records']}
    expected, records = [], []
    for ctx in inp['contexts']:
        rec = by_label[ctx['pair_id']]
        for side in ('left', 'right'):
            lab = rec[side]
            full = np.zeros(lab['length'], dtype=np.int8)
            full[lab['positive']] = 1
            # The dataset extracts observed residues in ascending sequence order.
            y = full[np.sort(lab['observed'])]
            expected.append(y)
            records.append({'pair_id': ctx['pair_id'], 'side': side, 'n': len(y)})
    require(ours['records'] == records and len(records) == 1024, 'Direction order mismatch')
    with np.load(args.ours / 'predictions.npz', allow_pickle=False) as raw:
        y, prob, lengths = raw['labels'], raw['probabilities'], raw['lengths']
    require(np.array_equal(y, np.concatenate(expected))
            and np.array_equal(lengths, [len(v) for v in expected])
            and len(y) == 289902 and prob.shape == y.shape
            and np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(),
            'Residue labels, coverage or probability alignment failed')
    cuts = np.r_[0, np.cumsum(lengths)]
    macro_ap = np.mean([average_precision_score(y[a:b], prob[a:b])
                        for a, b in zip(cuts[:-1], cuts[1:])])
    precision, recall, _ = precision_recall_curve(y, prob)
    metric = {
        'coverage_pair_contexts': 512, 'directional_contexts': 1024,
        'residues': len(y), 'positive_fraction': float(y.mean()),
        'macro_ap': float(macro_ap),
        'pooled_ap': float(average_precision_score(y, prob)),
        'pooled_pr_auc_trapezoid': float(auc(recall, precision)),
        'pooled_auroc': float(roc_auc_score(y, prob)),
        'official_0p5': {'gate': 0.5, 'f1': float(f1_score(y, prob > 0.5)),
                         'mcc': float(matthews_corrcoef(y, prob > 0.5))},
        'exact_ties_at_0p5': int(np.sum(prob == 0.5)),
    }
    models = {'ours_rcsb8m_epoch5': metric, **xpair['models']}
    for name, model in models.items():
        require(model['coverage_pair_contexts'] == 512 and model['directional_contexts'] == 1024
                and model['residues'] == len(y)
                and np.isclose(model['positive_fraction'], y.mean(), rtol=0, atol=1e-15),
                'Model coverage differs: ' + name)
    keys = ('macro_ap', 'pooled_ap', 'pooled_pr_auc_trapezoid', 'pooled_auroc')
    differences = {}
    for name in xpair['models']:
        differences[name] = {key: metric[key] - models[name][key] for key in keys}
        differences[name].update({key + '_at_0p5': metric['official_0p5'][key]
                                  - models[name]['official_0p5'][key] for key in ('f1', 'mcc')})
    report = {
        'scope': 'same_frozen_val512_not_test', 'input_sha256': INPUT_SHA,
        'labels_sha256': LABEL_SHA, 'ours_checkpoint_sha256': CHECKPOINT_SHA,
        'xpair_metrics_sha256': digest(args.xpair_metrics),
        'ours_predictions_sha256': digest(args.ours / 'predictions.npz'),
        'ours_summary_sha256': digest(args.ours / 'summary.json'),
        'contract': xpair['contract'], 'threshold_operator': '>',
        'new_threshold_search': False, 'models': models,
        'ours_minus_baseline': differences,
        'limitations': ['Validation pilot, not a sealed test.',
                        'Ours uses ESM-C and predicted monomer structure; X-PAIR uses sequences.',
                        'X-PAIR pretraining overlap audit pending; no significance claim.',
                        'Existing trusted labels reused; geometry not independently recomputed.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), 'Refuse to overwrite a comparison artifact')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
