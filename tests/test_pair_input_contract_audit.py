import importlib.util
from pathlib import Path

import torch
import pytest

SPEC = importlib.util.spec_from_file_location('input_audit', Path(__file__).parents[1]/
    'scripts/analysis/audit_pair_input_contract.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_tensor_digest_tracks_inputs_not_order():
    inputs = dict(esm=torch.arange(24).view(2, 3, 4), mask=torch.ones(2, 3, dtype=torch.bool))
    digest = AUDIT.tensor_digest(inputs)
    assert digest == AUDIT.tensor_digest(dict(reversed(list(inputs.items()))))
    assert digest == AUDIT.tensor_digest({k: v.clone() for k, v in inputs.items()})
    assert digest != AUDIT.tensor_digest(dict(inputs, esm=inputs['esm']+1))
    assert digest != AUDIT.tensor_digest(dict(inputs, esm=inputs['esm'].float()))


def test_probability_change_ignores_common_logit_shift():
    logits = torch.tensor([[.2, .9], [3., 2.]])
    assert AUDIT.max_probability_change(logits, logits+2) < 1e-6
    assert AUDIT.max_probability_change(logits, logits.flip(-1)) > .4


def test_padding_donor_expands_both_sides_and_excludes_identical_pair():
    def row(pid, left, right, a, b):
        return dict(pair_id=pid, left_seq_id=left, right_seq_id=right,
                    left_length=a, right_length=b)
    rows = [row('a', 'A', 'B', 100, 150), row('b', 'A', 'B', 200, 200),
            row('c', 'C', 'D', 120, 220), row('d', 'D', 'E', 160, 180)]
    assert AUDIT.padding_donor(rows, [0], 256) == 3
    with pytest.raises(ValueError, match='expands'):
        AUDIT.padding_donor(rows[:3], [0], 256)
