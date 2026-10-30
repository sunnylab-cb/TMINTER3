"""Evaluate the previously selected RCSB 8M checkpoint on frozen X-PAIR val512."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

INPUT_SHA = 'ee61e1cb3a7f7fa1213c4f8289cc6fa2a027a260f3778c7509b81af34d7897e2'
LABEL_SHA = 'd2f7e670f84cae102143782478d00d91fd0ed88d39680b4bb258b31960a6302c'
TRAIN_MANIFEST_SHA = '4d908b57ecba298697852d0ed808c9ed48989dfd77439db42b0f97f2f7d31fa8'
CHECKPOINT_SHA = '0dddf1302a02cb0330899d43db32a1bb4306302997ef65dc7470f9c5c06edbcd'


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for n in ('source', 'checkpoint', 'inputs', 'labels', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args()
    require(digest(a.inputs) == INPUT_SHA and digest(a.labels) == LABEL_SHA, 'Wrong frozen cohort')
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(a.source / 'src'), str(a.source / 'scripts/training')]
    import numpy as np
    import torch
    import train_esm_pair as trainer
    from ppcbind.models.esm_pair import ESMPairSiteClassifier
    from ppcbind.data.esm_pair_dataset import collate_esm_pair_features
    require(Path(trainer.__file__).resolve().is_relative_to(a.source.resolve()), 'Wrong frozen trainer source')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    require(torch.cuda.is_available(), 'Allocated GPU unavailable')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    require(digest(a.checkpoint) == CHECKPOINT_SHA, 'Not the preselected RCSB 8M checkpoint')
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    require(ckpt['epoch'] == 5, 'Expected preselected full-val-best epoch 5')
    config = ckpt['config']['config']
    require(config['data']['pair_data_backend'] == 'rcsb_observed', 'Wrong data backend')
    manifest = Path(config['data']['pair_manifest'])
    require(digest(manifest) == TRAIN_MANIFEST_SHA, 'Training split changed')
    inp = json.loads(a.inputs.read_text())
    labels = json.loads(a.labels.read_text())
    require(inp['test_evaluation'] is False and len(inp['contexts']) == 512
            and all(r['split'] == 'val' for r in inp['contexts']), 'Only val512 is permitted')
    by_label = {r['pair_id']: r for r in labels['records']}
    require(set(by_label) == {r['pair_id'] for r in inp['contexts']}, 'Label context mismatch')
    selected_uids, selected_pdbs = set(inp['sequences']), {r['pdb_id'].lower() for r in inp['contexts']}
    train_uids, train_pdbs = set(), set()
    with manifest.open() as f:
        for r in csv.DictReader(f):
            if r['split'] == 'train':
                train_uids.update(r[s + '_seq_id'] for s in ('left', 'right'))
                train_pdbs.add(r['pdb_id'].lower())
    require(not selected_uids & train_uids and not selected_pdbs & train_pdbs,
            'Selected validation contexts overlap training identities')
    print('BUILDING_FROZEN_VAL_LOADER', flush=True)
    dataset, loader = trainer._build_loader(config, 'val', False)
    del loader
    lookup = {r['pair_id']: i for i, r in enumerate(dataset.rows)}
    model = ESMPairSiteClassifier(config['model'], config['pair_model']).to(device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    chains, records = [], []
    started = time.time()
    with torch.inference_mode():
        for n, ctx in enumerate(inp['contexts'], 1):
            pid = ctx['pair_id']
            require(pid in lookup, 'Pair is not in frozen validation split')
            source_row = dataset.rows[lookup[pid]]
            require(source_row['pdb_id'] == ctx['pdb_id'] and source_row['split'] == 'val', 'Wrong context')
            item = dataset[lookup[pid]]
            for side in ('left', 'right'):
                lab = by_label[pid][side]
                seq = inp['sequences'][ctx[side + '_uid']]
                require(source_row[side + '_seq_id'] == ctx[side + '_uid'] == lab['uid']
                        and ''.join(item[side]['residue_names_1']) == seq, 'Feature sequence mismatch')
                expected = torch.full((len(seq),), -100, dtype=torch.long)
                expected[lab['observed']] = 0
                expected[lab['positive']] = 1
                require(torch.equal(item[side]['labels'], expected), 'Target differs from X-PAIR cohort')
            batch = collate_esm_pair_features([item])
            left, yl = trainer._move_side(batch['left'], device)
            right, yr = trainer._move_side(batch['right'], device)
            with torch.amp.autocast('cuda', enabled=bool(config['training'].get('amp', True))):
                out = model(left, right)
            for side, target in [('left', yl), ('right', yr)]:
                keep = target[0] != -100
                y = target[0][keep].cpu().numpy()
                prob = out[side + '_logits'][0][keep].float().softmax(-1)[:, 1].cpu().numpy()
                require(y.ndim == prob.ndim == 1 and len(y) == len(prob) and np.isfinite(prob).all(), 'Invalid score')
                chains.append((pid + ':' + side, prob, y))
                records.append({'pair_id': pid, 'side': side, 'n': len(y)})
            if n % 32 == 0 or n == 1:
                print(json.dumps({'completed_pairs': n, 'total': 512, 'seconds': time.time() - started}), flush=True)
    require(len(chains) == 1024, 'Incomplete directional coverage')
    metrics = trainer._summarize_predictions(chains, [], allow_threshold_search=False)
    np.savez_compressed(a.output / 'predictions.npz', labels=np.concatenate([r[2] for r in chains]),
                        probabilities=np.concatenate([r[1] for r in chains]),
                        lengths=np.array([len(r[2]) for r in chains]))
    report = {'scope': 'same_frozen_val512_not_test', 'checkpoint': str(a.checkpoint),
              'checkpoint_sha256': digest(a.checkpoint), 'epoch': 5,
              'selection': 'preselected by full native validation macro AP, not val512',
              'input_sha256': INPUT_SHA, 'labels_sha256': LABEL_SHA,
              'train_manifest_sha256': TRAIN_MANIFEST_SHA, 'test_evaluation': False,
              'train_overlap': {'exact_sequence': 0, 'pdb': 0},
              'amp': bool(config['training'].get('amp', True)), 'torch': torch.__version__,
              'normalization_stats_sha256': digest(Path(config['data']['rcsb_normalization_stats'])),
              'threshold_search': False,
              'metrics': metrics, 'records': records, 'seconds': time.time() - started,
              'peak_vram_bytes': torch.cuda.max_memory_allocated(device)}
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
