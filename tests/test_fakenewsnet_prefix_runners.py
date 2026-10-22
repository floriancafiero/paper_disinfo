"""Mock notebook execution: window propagation, cleanup and handoff behavior."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_virality_quantile_grid as grid
import colleague_revision_run as handoff


class RunnerTests(unittest.TestCase):
    def test_grid_passes_windows_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "colab_no_embeddings/outputs/fakenewsnet_virality_prefix_q50"
            target.mkdir(parents=True)
            for name in ('fakenewsnet_virality_merged_cv_results.csv', 'fakenewsnet_virality_merged_summary.csv'):
                (target / name).write_text('aggregate')

            def execute():
                self.assertEqual(os.environ['FNN_WINDOWS'], '1,10,full')
                self.assertEqual(os.environ['VIRALITY_QUANTILE'], '0.5')

            client = SimpleNamespace(execute=execute)
            modules = {'nbformat': SimpleNamespace(read=lambda *a, **kw: {}),
                       'nbclient': SimpleNamespace(NotebookClient=lambda *a, **kw: client)}
            with patch.object(grid, 'ROOT', root), patch.dict(sys.modules, modules), patch.dict(os.environ, {'FNN_WINDOWS': '3', 'VIRALITY_QUANTILE': '0.9'}):
                grid.run('fakenewsnet', [.5], '1,10,full')
                self.assertEqual(os.environ['FNN_WINDOWS'], '3')
                self.assertEqual(os.environ['VIRALITY_QUANTILE'], '0.9')

    def test_grid_restores_environment_on_kernel_error(self):
        def execute():
            raise RuntimeError('kernel failed')
        modules = {'nbformat': SimpleNamespace(read=lambda *a, **kw: {}),
                   'nbclient': SimpleNamespace(NotebookClient=lambda *a, **kw: SimpleNamespace(execute=execute))}
        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            os.environ.pop('FNN_WINDOWS', None)
            with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                grid.run('fakenewsnet', [.5], '1')
            self.assertNotIn('FNN_WINDOWS', os.environ)

    def test_handoff_propagates_windows(self):
        with patch.object(handoff, 'run') as run, patch.object(handoff, 'execute_notebook') as notebook, patch.object(handoff.Path, 'mkdir'):
            handoff.train('1,10,full')
        self.assertEqual(notebook.call_args.args[2], '1,10,full')
        self.assertEqual(run.call_args.args[0][-2:], ['--fnn-windows', '1,10,full'])

    def test_handoff_notebook_restores_environment(self):
        def execute():
            self.assertEqual(os.environ['FNN_WINDOWS'], '5')
        modules = {'nbformat': SimpleNamespace(read=lambda *a, **kw: {}),
                   'nbclient': SimpleNamespace(NotebookClient=lambda *a, **kw: SimpleNamespace(execute=execute))}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'FNN_WINDOWS': '3'}):
            handoff.execute_notebook('unused.ipynb', 'test', '5')
            self.assertEqual(os.environ['FNN_WINDOWS'], '3')


if __name__ == '__main__':
    unittest.main()
