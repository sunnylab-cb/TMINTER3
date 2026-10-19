"""Create a matched PDBTM S5 graph/no-graph training control."""
import argparse
import copy
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = yaml.safe_load(args.source.read_text())
    if not base['model']['use_contact_graph'] or base['training']['initialization'] != 'scratch':
        raise ValueError('Expected graph-enabled scratch S5 source')
    if base['training'].get('evaluate_test', True):
        raise ValueError('Test evaluation must remain disabled')
    args.output.mkdir(parents=True, exist_ok=False)
    for enabled, name in ((True, 'graph'), (False, 'no_graph')):
        cfg = copy.deepcopy(base)
        cfg['model']['use_contact_graph'] = enabled
        cfg['training']['output_dir'] = f'runs/pdbtm_S5_graph_control_{name}_seed49_20261018'
        cfg['metadata']['matrix_variant'] = f'Matched S5 graph control: {name}'
        cfg['metadata']['test_policy'] = 'Validation-only diagnostic training; no test evaluation'
        (args.output / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps({'source': str(args.source), 'output': str(args.output),
                      'changed_training_keys': ['output_dir'],
                      'changed_model_keys': ['use_contact_graph'], 'test_evaluation': False}))


if __name__ == '__main__':
    main()
