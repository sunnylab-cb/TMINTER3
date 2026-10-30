"""Official X-PAIR/Ankh inference, exact-sequence cache and unrounded residue outputs."""
import argparse
import fcntl
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import traceback
import zipfile

from check_runtime import ASSETS, SOURCE, load_checkpoint, require, torch_identity

REVISION = '74b371dbfa3ee0a05d32ae74df0c2e0b82d6b9a6'
WEIGHT_SHA = '517b6e8b279dedcb477af240b35c46bd6eb3307723eb281e60d4b2c8a87b889b'
INPUT_SHA = 'ee61e1cb3a7f7fa1213c4f8289cc6fa2a027a260f3778c7509b81af34d7897e2'
MANIFEST_SHA = '682a1815212000991f0ceb9ce2c784aa9085bebf8d0d41fa0e5436836eca4108'
WHEEL_SHA = '1bb6d7230300a492422e8f739925bb9e5dfdfc44d13af91c097cdba3ff611f06'
TOKENIZER_HASHES = {
    'config.json': '360abb5f6cc715fa8af9b4786706a71fe34c3bee587f86cef49186dcba5e7e84',
    'tokenizer.json': 'b4533f607d9fd665f2d9d94b0cf71870a6fc2fc2ae7cbd516d0e43a9efb406fd',
    'tokenizer_config.json': '41fcf5b9ace40b2e614c8c12775f033cbbc696871504729c92e578cb21323025',
    'special_tokens_map.json': 'c8995d2f8037fe3a8cfdef30475365e1c314c417880b75abaf8296e3c05d42d6',
}
CHECKPOINTS = [
    ('interface_pioneer', ASSETS / 'interface_pioneer.ckpt', 'interface',
     '7ab260dc7917205367cb56ba8f706b36ffa502540c44a5395a1a51457a4c6bd1'),
    ('multitask_xfair', SOURCE / 'xpair/default_model/multitask_xfair.ckpt', 'multitask',
     'd8c50d03cd6107d2b411d7095a34243ec1b31e6b7cc1d590329c9328c1e8df72'),
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        run(a)
    except Exception as exc:
        save_json(a.output / 'failure.json', {'status': 'failed', 'smoke': a.smoke,
                  'error_type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc()})
        raise


def run(a):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    torch.set_num_threads(4)
    identity = torch_identity(torch)
    device = torch.device(a.device)
    require(device.type == 'cuda' and torch.cuda.is_available(), 'GPU required for this pilot')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(a.output / 'run.log')])
    inp = json.loads(a.input.read_text())
    require(sha(a.input) == INPUT_SHA and inp['manifest_sha256'] == MANIFEST_SHA,
            'Not the frozen validation input snapshot')
    require(inp['scope'] == 'frozen_val512_label_free_inference' and inp['test_evaluation'] is False,
            'Only frozen validation inference is allowed')
    require(len(inp['contexts']) == 512 and all(r['split'] == 'val' for r in inp['contexts']),
            'Validation context contract changed')
    sequences = inp['sequences']
    require(len(sequences) == 532 and len(inp['pairs']) == 428, 'Validation cohort counts changed')
    require(all(uid == 'useq_' + hashlib.sha256(seq.encode()).hexdigest()
                for uid, seq in sequences.items()), 'Sequence hash mismatch')
    pairs = inp['pairs']
    for r in pairs:
        require(r['key'] == hashlib.sha256((r['left_uid'] + ':' + r['right_uid']).encode()).hexdigest(),
                'Pair key mismatch')
    wheel = ASSETS / 'xpair-0.1.1-py3-none-any.whl'
    require(sha(wheel) == WHEEL_SHA, 'Official source wheel changed')
    with zipfile.ZipFile(wheel) as z:
        expected_files = {n for n in z.namelist() if n.endswith(('.py', '.ckpt'))}
        actual_files = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.ckpt')}
        require(actual_files == expected_files, 'Official source file set changed')
        for name in z.namelist():
            if name.endswith(('.py', '.ckpt')):
                require((SOURCE / name).read_bytes() == z.read(name), 'Official source changed: ' + name)
    import xpair.model as official_model
    require(Path(official_model.__file__).resolve() == SOURCE / 'xpair/model.py', 'Unexpected model source')
    for name, digest in TOKENIZER_HASHES.items():
        require(sha(ASSETS / 'ankh-large' / name) == digest, 'Tokenizer/config changed: ' + name)
    for name, path, task, digest in CHECKPOINTS:
        require(sha(path) == digest, 'Checkpoint changed: ' + name)
        h = load_checkpoint(path, torch, np)['hyper_parameters']
        require(h['plm'] == 'ankh_large' and h['model_task'] == task
                and h['reduced_dim'] == 128 and h['num_cross_layers'] == 2 and h['num_heads'] == 2,
                'Unexpected checkpoint architecture: ' + name)
    if a.smoke:
        ordered = sorted(pairs, key=lambda r: (max(len(sequences[r['left_uid']]), len(sequences[r['right_uid']])), r['key']))
        pairs = [next(r for r in ordered if r['left_uid'] == r['right_uid']),
                 next(r for r in ordered if r['left_uid'] != r['right_uid'])]
    needed = sorted({r[s + '_uid'] for r in pairs for s in ('left', 'right')},
                    key=lambda u: (len(sequences[u]), u))
    cache = a.output.parent / 'embeddings_ankh_fp32'
    cache.mkdir(exist_ok=True)
    # Keep the file handle alive to serialize writers of the shared embedding cache.
    cache_lock = (cache / '.lock').open('a')
    fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    weights = ASSETS / 'ankh-large/pytorch_model.bin'
    logging.info('Verifying frozen Ankh weights')
    require(sha(weights) == WEIGHT_SHA, 'Ankh weight hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / 'ankh-large', local_files_only=True)

    def embedding(uid):
        d = torch.load(cache / (uid + '.pt'), map_location='cpu', weights_only=True)
        seq = sequences[uid]
        require(d['sequence'] == seq and d['revision'] == REVISION and d['weight_sha256'] == WEIGHT_SHA,
                'Embedding cache provenance mismatch')
        require(d['tokenizer_hashes'] == TOKENIZER_HASHES and d['dtype'] == 'float32'
                and d['normalized_sequence'] == re.sub(r'[UZOBJ]', 'X', seq.upper()),
                'Embedding tokenizer/normalization mismatch')
        x = d['embedding']
        require(x.dtype == torch.float32 and tuple(x.shape) == (len(seq), 1536)
                and torch.isfinite(x).all(), 'Invalid embedding')
        return x

    pending = []
    for uid in needed:
        if (cache / (uid + '.pt')).exists():
            embedding(uid)
        else:
            pending.append(uid)
    logging.info('Inputs=%d unique_sequences=%d pending_embeddings=%d smoke=%s',
                 len(pairs), len(needed), len(pending), a.smoke)
    start = time.time()
    if pending:
        plm = T5EncoderModel.from_pretrained(ASSETS / 'ankh-large', local_files_only=True,
                                           torch_dtype=torch.float32).eval().to(device).float()
        require(all(x.dtype == torch.float32 for x in plm.parameters()), 'Ankh dtype mismatch')
        for n, uid in enumerate(pending, 1):
            seq = sequences[uid]
            normalized = re.sub(r'[UZOBJ]', 'X', seq.upper())
            require(set(normalized) <= set('ACDEFGHIKLMNPQRSTVWYX'), 'Unsupported sequence alphabet: ' + uid)
            tokens = tokenizer(normalized, return_tensors='pt', add_special_tokens=True,
                               return_special_tokens_mask=True)
            keep = tokens['attention_mask'].bool() & ~tokens['special_tokens_mask'].bool()
            require(int(keep.sum()) == len(seq), 'Tokenizer/residue count mismatch')
            recovered = tokenizer.decode(tokens['input_ids'][keep], skip_special_tokens=True).replace(' ', '')
            require(recovered == normalized, 'Tokenizer residue order mismatch')
            t = time.time()
            with torch.inference_mode():
                x = plm(input_ids=tokens['input_ids'].to(device),
                        attention_mask=tokens['attention_mask'].to(device)).last_hidden_state[keep.to(device)].cpu()
            require(x.dtype == torch.float32 and tuple(x.shape) == (len(seq), 1536)
                    and torch.isfinite(x).all(), 'Invalid generated embedding')
            target = cache / (uid + '.pt')
            tmp = target.with_suffix('.tmp')
            torch.save({'sequence': seq, 'normalized_sequence': normalized, 'revision': REVISION,
                        'weight_sha256': WEIGHT_SHA, 'tokenizer_hashes': TOKENIZER_HASHES,
                        'dtype': 'float32', 'embedding': x}, tmp)
            tmp.replace(target)
            logging.info('EMBED %d/%d %s L=%d seconds=%.2f', n, len(pending), uid, len(seq), time.time() - t)
        del plm, x
        gc.collect()
        torch.cuda.empty_cache()
    summary = {'scope': inp['scope'], 'smoke': a.smoke, 'input_sha256': sha(a.input),
               'manifest_sha256': inp['manifest_sha256'], 'input_pairs': len(pairs),
               'unique_sequences': len(needed), 'test_evaluation': False,
               'labels_supplied': False, 'interaction_gating': False,
               'precision': 'float32_unrounded', 'models': {}, 'torch': identity,
               'tokenizer_hashes': TOKENIZER_HASHES, 'source_wheel_sha256': WHEEL_SHA,
               'ankh_revision': REVISION, 'ankh_weight_sha256': WEIGHT_SHA}
    for name, path, task, digest in CHECKPOINTS:
        ckpt = load_checkpoint(path, torch, np)
        model = official_model.XPairModel(ckpt['hyper_parameters']).eval().to(device).float()
        require(all(x.dtype == torch.float32 for x in model.parameters()), 'X-PAIR dtype mismatch')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        out = a.output / name
        out.mkdir(exist_ok=True)
        for n, r in enumerate(pairs, 1):
            u, v = r['left_uid'], r['right_uid']
            x1, x2 = embedding(u).unsqueeze(0).to(device), embedding(v).unsqueeze(0).to(device)
            batch = {'input1': (x1, torch.ones(x1.shape[:2], dtype=torch.bool, device=device)),
                     'input2': (x2, torch.ones(x2.shape[:2], dtype=torch.bool, device=device))}
            with torch.inference_mode():
                interaction, left, right = model(batch, task=task)
                require(left.dtype == torch.float32 and right.dtype == torch.float32, 'Logit dtype mismatch')
                left, right = torch.sigmoid(left[0]).cpu().numpy(), torch.sigmoid(right[0]).cpu().numpy()
            require(left.shape == (len(sequences[u]),) and right.shape == (len(sequences[v]),)
                    and np.isfinite(left).all() and np.isfinite(right).all(), 'Invalid residue scores')
            target = out / (r['key'] + '.npz')
            tmp = target.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, left=left, right=right, left_uid=u, right_uid=v,
                                    checkpoint_sha256=digest, input_sha256=summary['input_sha256'],
                                    ankh_revision=REVISION, ankh_weight_sha256=WEIGHT_SHA,
                                    interaction_probability=np.nan if interaction is None else float(torch.sigmoid(interaction[0]).item()))
            tmp.replace(target)
            logging.info('PAIR %s %d/%d L=%d/%d', name, n, len(pairs), len(left), len(right))
        summary['models'][name] = {'checkpoint_sha256': digest, 'completed': len(pairs)}
        del model, x1, x2
        gc.collect()
        torch.cuda.empty_cache()
    require(torch_identity(torch) == identity, 'Torch identity changed')
    summary.update(status='complete', seconds=time.time() - start,
                   peak_vram_bytes=torch.cuda.max_memory_allocated(device))
    save_json(a.output / 'summary.json', summary)
    logging.info('COMPLETE %s', json.dumps(summary))


if __name__ == '__main__':
    main()
