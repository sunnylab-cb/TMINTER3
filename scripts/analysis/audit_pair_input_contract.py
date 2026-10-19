"""Small validation-only frozen-checkpoint input/swap/padding probes."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch


def tensor_digest(inputs):
    digest = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        digest.update(key.encode())
        if value is None:
            digest.update(b'None')
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError('Unexpected non-tensor model input: ' + key)
        value = value.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def max_probability_change(a, b):
    return float((a.float().softmax(-1)[..., 1]-b.float().softmax(-1)[..., 1]).abs().max())


def padding_donor(rows, selected, max_length):
    groups = {tuple(sorted((rows[i]['left_seq_id'], rows[i]['right_seq_id']))) for i in selected}
    longest = max(int(rows[i][side+'_length']) for i in selected for side in ('left', 'right'))
    candidates = [i for i, row in enumerate(rows)
                  if min(int(row[s+'_length']) for s in ('left', 'right')) > longest
                  and max(int(row[s+'_length']) for s in ('left', 'right')) <= 2*max_length
                  and tuple(sorted((row['left_seq_id'], row['right_seq_id']))) not in groups]
    if not candidates:
        raise ValueError('No distinct validation donor expands both sides')
    return min(candidates, key=lambda i: (sum(int(rows[i][s+'_length'])
                                            for s in ('left', 'right')), rows[i]['pair_id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frozen-code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs-per-kind', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.pairs_per_kind <= 0 or args.max_length <= 0:
        raise ValueError('Positive cohort limits required')
    frozen = args.frozen_code.resolve()
    sys.path[:0] = [str(frozen/'src'), str(frozen/'scripts/training')]
    from train_esm_pair import _build_loader, _move_side
    from ppcbind.data.esm_pair_dataset import collate_esm_pair_features
    from ppcbind.models.esm_pair import ESMPairSiteClassifier
    objects = (_build_loader, _move_side, collate_esm_pair_features, ESMPairSiteClassifier)
    sources = {Path(inspect.getfile(obj)).resolve() for obj in objects}
    if any(not p.is_relative_to(frozen) for p in sources):
        raise ValueError('Imported live code instead of frozen deployment')
    torch.set_num_threads(2)
    torch.manual_seed(49)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']['config']
    if (config['data'].get('pair_data_backend') != 'rcsb_observed'
            or config['pair_model']['mode'] != 'outer_product'):
        raise ValueError('Expected RCSB outer-product checkpoint')
    dataset, _ = _build_loader(config, 'val', False)
    eligible = [i for i, row in enumerate(dataset.rows)
                if max(int(row[s+'_length']) for s in ('left', 'right')) <= args.max_length]
    if not eligible:
        raise ValueError('No eligible validation pairs')
    selected, seen = [], set()
    for homomer in (True, False):
        added = 0
        for index in sorted(eligible, key=lambda i: hashlib.sha256(
                ('inputprobe20261019'+dataset.rows[i]['pair_id']).encode()).digest()):
            row = dataset.rows[index]
            key = tuple(sorted((row['left_seq_id'], row['right_seq_id'])))
            if (key[0] == key[1]) != homomer or key in seen:
                continue
            selected.append(index)
            seen.add(key)
            added += 1
            if added == args.pairs_per_kind:
                break
        if added != args.pairs_per_kind:
            raise ValueError('Insufficient distinct validation sequence pairs')
    donor_index = padding_donor(dataset.rows, selected, args.max_length)
    donor = dataset[donor_index]
    model = ESMPairSiteClassifier(config['model'], config['pair_model']).to(device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    if not bool(model.adapter.site_residual[-1].weight.count_nonzero()):
        raise ValueError('Zero-initialized partner readout would make probe trivial')
    sources.add(Path(inspect.getfile(type(model.base))).resolve())
    sources.add(Path(inspect.getfile(type(dataset))).resolve())
    if any(not p.is_relative_to(frozen) for p in sources):
        raise ValueError('Imported non-frozen base or dataset code')
    records = []
    with torch.inference_mode():
        for index in selected:
            row = dataset.rows[index]
            item = dataset[index]
            batch = collate_esm_pair_features([item])
            left, _ = _move_side(batch['left'], device)
            right, _ = _move_side(batch['right'], device)
            hleft, hright = tensor_digest(left), tensor_digest(right)
            out = model(left, right)
            swap = model(right, left)
            repeat = model(left, right)
            combined = collate_esm_pair_features([item, donor])
            a, _ = _move_side(combined['left'], device)
            b, _ = _move_side(combined['right'], device)
            padded = model(a, b)
            record = dict(pair_id=row['pair_id'], pdb_id=row['pdb_id'],
                left_uid=row['left_seq_id'], right_uid=row['right_seq_id'],
                left_input_sha256=hleft, right_input_sha256=hright,
                homomer=row['left_seq_id'] == row['right_seq_id'], dtype='fp32',
                batch_partner_pair_id=dataset.rows[donor_index]['pair_id'])
            for side, other in (('left', 'right'), ('right', 'left')):
                length = int(row[side+'_length'])
                reference = out[side+'_logits'][0, :length]
                record[side+'_swap_max_abs_logit'] = float((reference-swap[other+'_logits'][0, :length]).abs().max())
                record[side+'_repeat_max_abs_probability'] = max_probability_change(reference, repeat[side+'_logits'][0, :length])
                record[side+'_batch_max_abs_probability'] = max_probability_change(reference, padded[side+'_logits'][0, :length])
                record[side+'_length'] = length
                record[side+'_combined_width'] = int(combined[side]['protein_mask'].shape[1])
                if record[side+'_combined_width'] <= length:
                    raise ValueError('Padding probe did not expand tested side')
            if record['homomer']:
                if hleft != hright:
                    raise ValueError('Same-UID left/right actual input tensors differ')
                record['homomer_sides_max_abs_probability'] = max_probability_change(
                    out['left_logits'], out['right_logits'])
            records.append(record)
            print(json.dumps(record), flush=True)
    result = dict(split='val', test_labels_read=False, epoch=ckpt['epoch'],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_path=str(args.checkpoint), config=config,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        torch_version=torch.__version__, device=str(device), records=records,
        scope='Eight short validation sequence-pair groups by default; fp32 eval-mode probe, not an exhaustive or AMP determinism proof.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
