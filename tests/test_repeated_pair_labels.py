import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/audit_repeated_pair_labels.py'


class RepeatedPairLabelsTest(unittest.TestCase):
    def run_fixture(self, root, invalid=False, split='val', invalid_observed=False,
                    explicit_observed=True, context_detail=False):
        rows = []
        with (root / 'labels.jsonl').open('wb') as f:
            for i, (lp, rp) in enumerate([([0], [1]), ([0], [0])]):
                row = dict(pair_id=f'p{i}', pdb_id='demo', assembly_id='1', split=split,
                    left_seq_id='same', right_seq_id='same', left_chain_instance_id=f'A{i}',
                    right_chain_instance_id=f'B{i}', left_length=4, right_length=4,
                    left_observed=4, right_observed=4, left_positive=1, right_positive=1,
                    n_contacts=1, label_path='labels.jsonl', label_offset=f.tell())
                label = dict(row, contract='observed-polymer-heavy-atom-5p5-assembly-v2',
                    observed_sequence_only=True, heavy_only=True, cutoff_angstrom=5.5)
                label.update(left_positive=lp, right_positive=rp, residue_contacts=[[lp[0], rp[0]]])
                if explicit_observed:
                    label.update(left_observed_indices=list(range(4)), right_observed_indices=list(range(4)))
                if invalid:
                    label['right_positive'] = [5]
                if invalid_observed:
                    label['right_observed_indices'] = [0, 1, 2, 2]
                f.write((json.dumps(label)+'\n').encode())
                rows.append(row)
        with (root / 'manifest.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            excluded = dict(rows[0], pair_id='excluded_test', split='test',
                            label_path='must_not_be_opened.jsonl')
            writer.writerow(excluded)
        return subprocess.run([sys.executable, str(SCRIPT), '--manifest', str(root/'manifest.csv'),
            '--label-root', str(root), '--output', str(root/'output'), '--split', split]
            + (['--context-detail'] if context_detail else []),
            capture_output=True, text=True)

    def test_directional_homomer_patterns(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self.run_fixture(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['ordered_sequence_groups'], 1)
            self.assertEqual(summary['directional_records'], 4)
            self.assertEqual(summary['distinct_input_label_combinations'], 2)
            self.assertEqual(summary['records_in_varying_groups'], 4)
            example = summary['examples'][0]
            self.assertEqual(example['pattern_frequencies'], [3, 1])
            self.assertEqual(example['variable_residue_positions'], 2)
            self.assertEqual(example['jaccard'], 0)

    def test_invalid_positions_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self.run_fixture(root, invalid=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root/'output/summary.json').exists())

    def test_training_scope_excludes_test_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self.run_fixture(root, split='train')
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['split'], 'train')
            self.assertEqual(summary['pairs'], 2)
            self.assertFalse(summary['test_read'])
            self.assertIn('train_rows_sha256', summary)
            self.assertNotIn('val_rows_sha256', summary)

    def test_observed_indices_not_just_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self.run_fixture(root, invalid_observed=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root/'output/summary.json').exists())

    def test_frozen_all_observed_encoding_and_context_diversity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = self.run_fixture(root, explicit_observed=False, context_detail=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['observed_encoding_counts'], {'all_observed_contract': 4})
            self.assertEqual(summary['homomer_same_pair_sides']['different_masks'], 1)
            self.assertEqual(summary['homomer_same_pair_sides']['mean_jaccard'], .5)
            levels = summary['context_levels']
            self.assertEqual(levels['within_assembly_fixed_side']['groups'], 2)
            self.assertEqual(levels['within_assembly_fixed_side']['varying_groups'], 1)
            stats = summary['mask_effect_size']['all_groups']
            self.assertAlmostEqual(stats['expected_distinct_record_disagreement']['group_macro_mean'], .25)
            self.assertAlmostEqual(stats['mean_jaccard_to_modal_mask']['group_macro_mean'], .75)


if __name__ == '__main__':
    unittest.main()
