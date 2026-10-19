"""Descriptive strata from validation-only directional exports; no threshold tuning."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(rows):
    if not rows:
        return {'n': 0}
    unique = defaultdict(list)
    for r in rows:
        unique[r['sequence_pair']].append(r['ap'])
    return {'n': len(rows), 'pdbs': len({r['pdb_id'] for r in rows}),
            'unique_sequence_pairs': len(unique),
            'macro_ap': mean(r['ap'] for r in rows),
            'unique_pair_macro_ap': mean(mean(v) for v in unique.values()),
            'macro_prevalence': mean(r['positive_fraction'] for r in rows),
            'macro_ap_minus_prevalence': mean(r['ap']-r['positive_fraction'] for r in rows),
            'macro_ce': mean(r['ce'] for r in rows),
            'macro_brier': mean(r['brier'] for r in rows),
            'macro_probability_bias': mean(r['mean_probability']-r['positive_fraction'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    provenance = json.loads((args.input.parent / 'summary.json').read_text())
    if (provenance.get('complete') is not True or provenance.get('split') != 'val'
            or provenance.get('test_read') is not False):
        raise ValueError('Requires completed validation-only export')
    with args.input.open() as f:
        rows = list(csv.DictReader(f))
    identities = [(r['pair_id'], r['side']) for r in rows]
    pairs = {r['pair_id'] for r in rows}
    if (not rows or len(rows) != provenance['directional_records']
            or len(pairs) != provenance['pairs'] or len(identities) != len(set(identities))
            or set(identities) != {(p, s) for p in pairs for s in ('left', 'right')}):
        raise ValueError('Incomplete or duplicate directional records')
    groups = defaultdict(list)
    for r in rows:
        for k in ('ap', 'positive_fraction', 'ce', 'brier', 'mean_probability'):
            r[k] = float(r[k])
            if not math.isfinite(r[k]) or r[k] < 0 or (k != 'ce' and r[k] > 1):
                raise ValueError('Invalid metric: ' + k)
        p, length, mult = r['positive_fraction'], int(r['sequence_length']), int(r['pair_multiplicity'])
        bins = ['all', 'identity/' + ('same' if int(r['identical_sequence_pair']) else 'different'),
                'prevalence/' + ('zero' if p == 0 else '<=0.05' if p <= .05 else '<=0.10' if p <= .1 else '<=0.20' if p <= .2 else '>0.20'),
                'length/' + ('<100' if length < 100 else '100-299' if length < 300 else '300-599' if length < 600 else '600-999' if length < 1000 else '>=1000'),
                'multiplicity/' + ('1' if mult == 1 else '2-5' if mult <= 5 else '6-20' if mult <= 20 else '>20')]
        for group in bins:
            groups[group].append(r)
    output = {'source': str(args.input), 'checkpoint_sha256': provenance['checkpoint_sha256'],
              'epoch': provenance['epoch'], 'groups': {k: summarize(v) for k, v in groups.items()},
              'caution': 'Descriptive native-validation strata, not matched causal comparisons. AP minus prevalence is only a heuristic prevalence adjustment. Unique-pair averages retain differing structural-context labels. Same sequence is not a full biological homomer classification.'}
    with args.output.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
