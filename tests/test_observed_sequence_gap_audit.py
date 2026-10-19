import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('gap_audit', Path(__file__).parents[1]/
    'scripts/analysis/audit_observed_sequence_gaps.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture():
    digest = hashlib.sha256(b'ACDE').hexdigest()
    source = dict(sequence='ACDE', seq_sha256=digest, label_seq_gaps=[[2, 5]],
        residue_map=[dict(index=i, label_seq_id=p, insertion_code='A' if i == 1 else '')
                     for i, p in enumerate([1, 2, 5, 6])])
    return source, 'useq_'+digest


def test_internal_polymer_gap_preserves_insertion_residue():
    source, uid = fixture()
    result = AUDIT.gap_metrics(source, uid, 4)
    assert result == dict(length=4, internal_gap_count=1, internal_skipped_polymer_positions=2,
                          residues_with_insertion_code=1)


def test_reject_inconsistent_mapping_and_gap_metadata():
    source, uid = fixture()
    source['label_seq_gaps'] = []
    with pytest.raises(ValueError, match='gaps'):
        AUDIT.gap_metrics(source, uid, 4)
    source, uid = fixture()
    with pytest.raises(ValueError, match='hash'):
        AUDIT.gap_metrics(source, 'useq_bad', 4)
