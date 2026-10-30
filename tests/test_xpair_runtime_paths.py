"""Asset relocation must not import torch or change its installation."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmarks/xpair/check_runtime.py'


class RuntimePathTests(unittest.TestCase):
    def test_explicit_external_paths_without_loading_torch(self):
        with tempfile.TemporaryDirectory() as root:
            paths = {name: str(Path(root) / folder) for name, folder in (
                ('XPAIR_ASSETS', 'assets'), ('XPAIR_SOURCE', 'source'),
                ('XPAIR_BASE_TORCH', 'existing_torch'))}
            before = 'torch' in sys.modules
            with patch.dict(os.environ, paths), patch.object(sys, 'path', sys.path.copy()):
                namespace = runpy.run_path(str(SCRIPT), run_name='xpair_path_test')
                for key, env in [('ASSETS', 'XPAIR_ASSETS'), ('SOURCE', 'XPAIR_SOURCE'),
                                 ('BASE_TORCH', 'XPAIR_BASE_TORCH')]:
                    self.assertEqual(namespace[key], Path(paths[env]).resolve())
            self.assertEqual('torch' in sys.modules, before)

    def test_original_layout_is_preserved_from_asset_root(self):
        with tempfile.TemporaryDirectory() as root:
            assets = Path(root) / 'assets/X-PAIR'
            env = {k: v for k, v in os.environ.items()
                   if k not in {'XPAIR_ASSETS', 'XPAIR_SOURCE', 'XPAIR_BASE_TORCH'}}
            env['XPAIR_ASSETS'] = str(assets)
            with patch.dict(os.environ, env, clear=True), patch.object(sys, 'path', sys.path.copy()):
                namespace = runpy.run_path(str(SCRIPT), run_name='xpair_layout_test')
                self.assertEqual(namespace['ASSETS'], assets.resolve())
                self.assertEqual(namespace['SOURCE'], (Path(root) / 'official/X-PAIR-0.1.1').resolve())


if __name__ == '__main__':
    unittest.main()
