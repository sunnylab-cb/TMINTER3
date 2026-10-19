"""Validation-only internal polymer-coordinate gaps in frozen observed sequences."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path


def gap_metrics(source, uid, length):
    sequence = source['sequence']
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    mapping = source['residue_map']
    if (uid != 'useq_'+digest or source['seq_sha256'] != digest or len(sequence) != length
            or [r['index'] for r in mapping] != list(range(length))):
        raise ValueError('Observed-sequence hash or residue mapping mismatch')
    ids = [r['label_seq_id'] for r in mapping]
    if any(type(i) is not int for i in ids) or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError('Invalid polymer residue numbering')
    gaps = [[a, b] for a, b in zip(ids, ids[1:]) if b != a+1]
    if gaps != source['label_seq_gaps']:
        raise ValueError('Stored gaps disagree with residue mapping')
    return dict(length=length, internal_gap_count=len(gaps),
        internal_skipped_polymer_positions=sum(b-a-1 for a, b in gaps),
        residues_with_insertion_code=sum(bool(r['insertion_code']) for r in mapping))


def audit_entry(task):
    path, pdb, sources = task
    with gzip.open(path, 'rt') as handle:
        entry = json.load(handle)
    if entry['pdb_id'] != pdb or entry['contract'] != 'observed-polymer-heavy-atom-5p5-assembly-v2':
        raise ValueError('Unexpected source entry')
    return [dict(pdb_id=pdb, source_asym=asym, uid=info['uid'],
                 directional_records=info['records'], chain_instances=info['chains'],
                 **gap_metrics(entry['sources'][asym], info['uid'], info['length']))
            for asym, info in sorted(sources.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-mapping', type=Path, required=True)
    parser.add_argument('--expected-mapping-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1:
        raise ValueError('Fresh output directory and positive worker count required')
    digest = hashlib.sha256()
    with args.source_mapping.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != args.expected_mapping_sha256:
        raise ValueError('Source mapping differs from frozen materialization provenance')
    expected, appearances, selected_digest = {}, Counter(), hashlib.sha256()
    pdb_splits = defaultdict(set)
    with args.manifest.open(newline='') as handle:
        for row in csv.DictReader(handle):
            pdb_splits[row['split']].add(row['pdb_id'])
            if row['split'] != 'val':
                continue
            selected_digest.update((json.dumps(row, sort_keys=True)+'\n').encode())
            for side in ('left', 'right'):
                chain = row[side+'_chain_instance_id']
                info = (row['pdb_id'], row[side+'_seq_id'], int(row[side+'_length']))
                if expected.setdefault(chain, info) != info:
                    raise ValueError('Inconsistent chain-instance metadata')
                appearances[chain] += 1
    if not expected:
        raise ValueError('Empty validation cohort')
    if pdb_splits['val'] & (pdb_splits['train'] | pdb_splits['test']):
        raise ValueError('Validation PDB overlaps train/test: cannot open whole-entry sources')
    tasks, seen = {}, set()
    with args.source_mapping.open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            chain = row['chain_instance_id']
            if chain not in expected:
                continue
            if chain in seen:
                raise ValueError('Duplicate mapped chain')
            seen.add(chain)
            pdb, uid, length = expected[chain]
            if (row['pdb_id'], row['unique_id']) != (pdb, uid):
                raise ValueError('Manifest/source mapping mismatch')
            key = (row['tag_path'], pdb)
            sources = tasks.setdefault(key, {})
            info = sources.setdefault(row['source_label_asym_id'],
                dict(uid=uid, length=length, records=0, chains=0))
            if (info['uid'], info['length']) != (uid, length):
                raise ValueError('Inconsistent asymmetric-unit source')
            info['records'] += appearances[chain]
            info['chains'] += 1
    if seen != set(expected):
        raise ValueError('Validation chain mapping is incomplete')
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for index, records in enumerate(pool.map(audit_entry,
                [(path, pdb, sources) for (path, pdb), sources in sorted(tasks.items())], chunksize=16)):
            rows.extend(records)
            if (index+1) % 1000 == 0:
                print(json.dumps(dict(entries=index+1, total=len(tasks))), flush=True)
    uid_gap = defaultdict(set)
    for row in rows:
        uid_gap[row['uid']].add(row['internal_gap_count'] > 0)
    summary = dict(split='val', test_labels_read=False, source_entries=len(tasks),
        validation_pdb_overlap_with_train_test=0,
        source_read_scope='Whole JSON entries for validation PDBs only; manifest confirms zero train/test PDB overlap. Unselected asym sources in a validation PDB may be decoded but are not analyzed.',
        selected_manifest_rows_sha256=selected_digest.hexdigest(),
        source_mapping_sha256=digest.hexdigest(), sources=len(rows),
        sources_with_internal_gap=sum(r['internal_gap_count'] > 0 for r in rows),
        unique_uids=len(uid_gap), uids_with_any_internal_gap=sum(True in v for v in uid_gap.values()),
        uids_with_mixed_gap_presence=sum(len(v) > 1 for v in uid_gap.values()),
        directional_records=sum(r['directional_records'] for r in rows),
        directional_records_with_internal_gap=sum(r['directional_records'] for r in rows if r['internal_gap_count']),
        chain_instances=sum(r['chain_instances'] for r in rows),
        chain_instances_with_internal_gap=sum(r['chain_instances'] for r in rows if r['internal_gap_count']),
        internal_skipped_polymer_positions_by_source=sum(r['internal_skipped_polymer_positions'] for r in rows),
        sources_with_insertion_code=sum(r['residues_with_insertion_code'] > 0 for r in rows),
        caution='Internal gaps only. No terminal missingness or full construct length estimate; no causal attribution to performance. Counting source_asym, chain instance, UID and directional records are different aggregations.')
    if summary['directional_records'] != sum(appearances.values()):
        raise ValueError('Directional coverage mismatch')
    args.output.mkdir(parents=True)
    with (args.output/'sources.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
