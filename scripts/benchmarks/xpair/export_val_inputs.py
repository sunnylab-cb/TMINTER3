"""Export only observed validation sequences/pair identities; no interface labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    require(digest == '682a1815212000991f0ceb9ce2c784aa9085bebf8d0d41fa0e5436836eca4108',
            'Not the frozen val512 manifest')
    rows = list(csv.DictReader(a.manifest.open()))
    require(len(rows) == 512 and all(r['split'] == 'val' for r in rows), 'Invalid validation cohort')
    needed = {r[s + '_seq_id'] for r in rows for s in ('left', 'right')}
    sequences = {}
    with a.catalog.open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            uid = r['unique_id']
            if uid in needed:
                require(uid not in sequences, 'Duplicate catalog ID')
                seq = r['sequence']
                require(uid == 'useq_' + hashlib.sha256(seq.encode()).hexdigest(), 'UID mismatch')
                require(len(seq) == int(r['n_residues']), 'Catalog length mismatch')
                sequences[uid] = seq
    require(set(sequences) == needed, 'Missing catalog sequence')
    pairs, contexts = {}, []
    for r in rows:
        uids = [r[s + '_seq_id'] for s in ('left', 'right')]
        for s, uid in zip(('left', 'right'), uids):
            seq = sequences[uid]
            require(len(seq) == int(r[s + '_length']), 'Manifest length mismatch')
            require(hashlib.sha1(seq.encode()).hexdigest() == r[s + '_sequence_sha1'], 'Manifest SHA1 mismatch')
        key = hashlib.sha256((':'.join(uids)).encode()).hexdigest()
        pairs[key] = {'key': key, 'left_uid': uids[0], 'right_uid': uids[1]}
        contexts.append({'pair_id': r['pair_id'], 'pdb_id': r['pdb_id'], 'split': 'val',
                         'input_key': key, 'left_uid': uids[0], 'right_uid': uids[1]})
    require(len({r['pair_id'] for r in contexts}) == 512, 'Duplicate context')
    out = {'scope': 'frozen_val512_label_free_inference', 'test_evaluation': False,
           'manifest_sha256': digest, 'sequences': sequences,
           'pairs': sorted(pairs.values(), key=lambda x: x['key']), 'contexts': contexts}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    require(not a.output.exists(), 'Refuse to overwrite frozen input')
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'contexts': len(contexts), 'inputs': len(pairs), 'sequences': len(sequences),
                      'sha256': hashlib.sha256(a.output.read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
