"""CPU complete descriptors from exact-sequence Protenix monomers; no bridge."""
import csv
import hashlib
import json
import os
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import fcntl
import gemmi
import torch

MODULE = Path('/home/husrcf/Code/PPC/scripts/preparations/complete')
ROOT = Path('/data1/wholeset/RCSBprot/protenix')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(d, seq):
    require(d['sequence'] == seq and d['source'] == 'protenix_mini_esm_v0.5.0', 'Source/sequence mismatch')
    require(d['sequence_sha256'] == hashlib.sha256(seq.encode()).hexdigest(), 'Hash mismatch')
    for key, shape in [('spatial_scalar_raw_features', (len(seq), 89)),
                       ('physchem_features', (len(seq), 62)),
                       ('spatial_vector_features', (len(seq), 8, 3))]:
        require(tuple(d[key].shape) == shape, 'Shape mismatch: ' + key)
    require(all(torch.isfinite(v).all() for v in d.values() if isinstance(v, torch.Tensor)), 'Nonfinite features')


def worker(task):
    uid, seq, cif, output = task
    path = Path(output) / uid[5:7] / (uid + '.pt')
    try:
        import freesasa  # Required; never permit silent zero-SASA fallback.
        sys.path.insert(0, str(MODULE))
        from protein_features_v4 import extract_complete_features_v4
        torch.set_num_threads(1)
        if path.exists():
            try:
                d = torch.load(path, map_location='cpu', weights_only=False)
                validate(d, seq)
                return dict(unique_id=uid,status='skip',feature_path=str(path))
            except Exception:
                import time
                os.replace(path, path.with_suffix('.invalid.' + str(time.time_ns())))
        st=gemmi.read_structure(cif)
        require(len(st)==1 and len(st[0])==1, 'Expected single model and chain')
        observed=''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in st[0][0])
        require(observed == seq and hashlib.sha256(seq.encode()).hexdigest() == uid[5:], 'Observed sequence mismatch')
        residue_ids=[dict(number=r.seqid.num,insertion_code=r.seqid.icode) for r in st[0][0]]
        st[0][0].name='A'
        for i,r in enumerate(st[0][0],1): r.seqid=gemmi.SeqId(i,' ')
        with tempfile.TemporaryDirectory(prefix='protenix_complete_') as tmp:
            pdb=Path(tmp)/'input.pdb'; st.write_pdb(str(pdb))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                d=extract_complete_features_v4(pdb,mode='complex',normalization_stats=None)
        require(not any('DSSP failed' in str(w.message) for w in caught), 'DSSP failed')
        require(''.join(gemmi.find_tabulated_residue(r).one_letter_code.upper() for r in d['residue_names'])==seq, 'Feature residue mismatch')
        d.update(unique_id=uid,sequence=seq,sequence_sha256=uid[5:],
                 source='protenix_mini_esm_v0.5.0',source_cif=cif,source_residue_ids=residue_ids,
                 normalization='raw; train-only fitting deferred',freesasa_available=True)
        validate(d, seq)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp'); torch.save(d,temp); os.replace(temp,path)
        return dict(unique_id=uid,status='ok',feature_path=str(path),warnings=[str(w.message) for w in caught])
    except Exception as e:
        return dict(unique_id=uid,status='error',error=repr(e))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    import shutil,freesasa
    require(shutil.which('mkdssp'), 'DSSP required')
    sys.path.insert(0, str(MODULE))
    from protein_features_v4 import extract_complete_features_v4
    out=ROOT/'features/complete_raw_v1'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={}
    for ledger in sorted((ROOT/'runs').glob('**/progress.jsonl')):
        for line in ledger.open():
            if not line.endswith('\n'):break
            r=json.loads(line)
            if r.get('status')=='ok' and r.get('cif') and Path(r['cif']).is_file():sources[r['unique_id']]=r['cif']
    for directory in (ROOT/'transfers').glob('*'):
        for name in ('retained.jsonl','retained_on_star.jsonl'):
            index=directory/name
            if index.exists():
                for line in index.open():
                    r=json.loads(line)
                    if Path(r['cif']).is_file():sources[r['unique_id']]=r['cif']
    # rsync publishes each final filename by rename; partial files are elsewhere.
    for cif in (ROOT/'imports/hpc3/data').glob('*/*.cif'):
        if cif.with_name(cif.stem+'.summary_confidence.json').exists():sources.setdefault(cif.stem,str(cif))
    manifest=Path('/data1/wholeset/RCSBprot/sequences/observed_full_success_v1/unique_sequences.tsv')
    tasks=[]
    for r in csv.DictReader(manifest.open(),delimiter='\t'):
        if r['unique_id'] in sources:
            tasks.append((r['unique_id'],r['sequence'],sources[r['unique_id']],str(out/'pt')))
    tasks.sort()
    if args.limit: tasks=tasks[:args.limit]
    counts={}; print(json.dumps(dict(selected=len(tasks),workers=args.workers)),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (out/'progress.jsonl').open('a',buffering=1) as log:
        for i,result in enumerate(pool.map(worker,tasks,chunksize=1),1):
            log.write(json.dumps(result)+'\n')
            counts[result['status']]=counts.get(result['status'],0)+1
            if i%100==0 or i==len(tasks):
                summary=dict(total=len(tasks),processed=i,counts=counts)
                temp=out/'summary.tmp';temp.write_text(json.dumps(summary,indent=2));os.replace(temp,out/'summary.json')
                print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
