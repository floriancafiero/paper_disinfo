"""Backend contracts, propagation and opt-in real CUDA integration."""
import ast
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import baseline_backend as backend
import colleague_revision_run as handoff
import run_virality_quantile_grid as grid
from benchmark_baseline_backends import sample_indices
from fakenewsnet_prefix import make_splits


class BackendTests(unittest.TestCase):
    def test_cpu_original_parameters_without_gpu_imports(self):
        with patch.dict(sys.modules, {'cuml': None, 'cupy': None}):
            models = backend.build_baselines('cpu', record=False)
            self.assertEqual(models['rf'].n_estimators, 250)
            self.assertEqual(models['rf'].random_state, 42)
            self.assertEqual(models['rf'].n_jobs, -1)
            self.assertEqual(models['logreg'].max_iter, 1000)
            self.assertIsNone(models['logreg'].random_state)
            models = backend.build_baselines('cpu', logreg_random_state=42, record=False)
            self.assertEqual(models['logreg'].random_state, 42)

    def test_selection_and_environment_restore(self):
        with patch.dict(os.environ, {'BASELINE_BACKEND': 'cpu'}):
            with backend.backend_environment('gpu'):
                self.assertEqual(backend.resolve_backend(), 'gpu')
            self.assertEqual(backend.resolve_backend(), 'cpu')
        with self.assertRaises(ValueError):
            backend.resolve_backend('automatic')
        with patch.dict(sys.modules, {'cuml': None, 'cupy': None}):
            with self.assertRaisesRegex(RuntimeError, 'GPU baseline requested'):
                backend.build_baselines('gpu', record=False)

    def test_positive_class_column_and_validation(self):
        model = SimpleNamespace(classes_=np.array([1, 0]),
                                predict_proba=lambda X: np.array([[.8, .2], [.3, .7]]))
        np.testing.assert_allclose(backend.positive_probability(model, np.zeros((2, 1))), [.8, .3])
        model.predict_proba = lambda X: np.full((2, 2), np.nan)
        with self.assertRaisesRegex(ValueError, 'Invalid predicted'):
            backend.positive_probability(model, np.zeros((2, 1)))

    def test_reproducible_sample_shared_between_encoders(self):
        y = np.arange(12000) % 2
        sample = sample_indices(y)
        np.testing.assert_array_equal(sample, sample_indices(y))
        self.assertEqual(len(np.unique(sample)), 10000)
        self.assertEqual(y[sample].sum(), 5000)

    def test_handoff_passes_backend_kernel_and_restores_environment(self):
        with patch.object(handoff, 'require_gpu'), patch.object(handoff, 'run') as run, patch.object(handoff, 'execute_notebook') as notebook, patch.object(handoff.Path, 'mkdir'), patch.dict(os.environ, {'BASELINE_BACKEND': 'cpu'}):
            def check_env(*args, **kwargs):
                self.assertEqual(os.environ['BASELINE_BACKEND'], 'gpu')
            run.side_effect = check_env
            handoff.train('1,full', 'gpu', 'paper_disinfo')
            self.assertEqual(os.environ['BASELINE_BACKEND'], 'cpu')
            self.assertEqual(notebook.call_args.kwargs, {'baseline_backend': 'gpu', 'kernel_name': 'paper_disinfo'})
            self.assertIn('--baseline-backend', run.call_args.args[0])
            self.assertIn('paper_disinfo', run.call_args.args[0])

    def test_kernel_receives_backend_and_restores_on_failure(self):
        def client(*args, **kwargs):
            self.assertEqual(kwargs['kernel_name'], 'paper_disinfo')
            def execute():
                self.assertEqual(os.environ['BASELINE_BACKEND'], 'gpu')
                raise RuntimeError('test kernel failure')
            return SimpleNamespace(execute=execute)
        modules = {'nbformat': SimpleNamespace(read=lambda *a, **kw: {}),
                   'nbclient': SimpleNamespace(NotebookClient=client)}
        with patch.dict(sys.modules, modules), patch.object(grid, 'require_gpu'), patch.dict(os.environ, {'BASELINE_BACKEND': 'cpu'}):
            with self.assertRaisesRegex(RuntimeError, 'test kernel failure'):
                grid.run('fakenewsnet', [.5], '1', 'gpu', 'paper_disinfo')
            self.assertEqual(os.environ['BASELINE_BACKEND'], 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'test kernel failure'):
                handoff.execute_notebook('unused', 'test', '1', 'gpu', 'paper_disinfo')
            self.assertEqual(os.environ['BASELINE_BACKEND'], 'cpu')


@unittest.skipUnless(os.environ.get('RUN_GPU_TESTS') == '1', 'Set RUN_GPU_TESTS=1 for real CUDA tests')
class GPUIntegrationTests(unittest.TestCase):
    def assert_csv_contract(self, cpu, gpu):
        import pandas as pd
        self.assertEqual(list(cpu.columns), list(gpu.columns))
        self.assertEqual(len(cpu), len(gpu))
        for field in ('model', 'fold', 'n_fit', 'n_calibration'):
            self.assertEqual(cpu[field].tolist(), gpu[field].tolist())
        for frame in (cpu, gpu):
            restored = pd.read_csv(StringIO(frame.to_csv(index=False)))
            self.assertEqual(list(restored.columns), list(frame.columns))
            self.assertEqual(len(restored), len(frame))
            self.assertTrue(np.isfinite(restored[['auc', 'f1', 'f1_tuned']]).all().all())

    def test_cpu_gpu_evons_baseline_schema(self):
        import pandas as pd
        import torch
        from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
        path = ROOT / 'colab_no_embeddings/evons_disinformation_detection.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run_baselines')
        # Execute the real function without the module's top-level full training.
        ns = dict(torch=torch, pd=pd, HAS_XGB=False,
                  StratifiedShuffleSplit=StratifiedShuffleSplit, train_test_split=train_test_split,
                  fit_evaluate=backend.fit_evaluate, resolve_backend=backend.resolve_backend,
                  build_baselines=lambda **kw: backend.build_baselines(record=False, **kw))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        X = torch.from_numpy(np.random.default_rng(42).normal(size=(60, 8)).astype('float32'))
        y = np.arange(60) % 2
        frames = []
        for selected in ('cpu', 'gpu'):
            with backend.backend_environment(selected):
                frames.append(ns['run_baselines'](X[:, :4], X[:, 4:], y, n_splits=1))
        self.assert_csv_contract(*frames)

    def test_cpu_gpu_notebook_baseline_schema(self):
        import pandas as pd
        import torch
        X = torch.from_numpy(np.random.default_rng(42).normal(size=(60, 8)).astype('float32'))
        y = np.arange(60) % 2
        splits = make_splits(y, n_splits=1)
        for task in ('disinformation_detection', 'virality_prediction'):
            notebook = json.loads((ROOT / f'colab_no_embeddings/fakenewsnet_{task}.ipynb').read_text())
            source = ''.join(notebook['cells'][3]['source'])
            source = source[source.index('def run_baselines_cv'):]
            frames = []
            for selected in ('cpu', 'gpu'):
                ns = dict(np=np, pd=pd, HAS_XGB=False, mode='bert', window='1',
                          fit_evaluate=backend.fit_evaluate, resolve_backend=backend.resolve_backend,
                          split_fingerprint=__import__('fakenewsnet_prefix').split_fingerprint,
                          build_baselines=lambda **kw: backend.build_baselines(record=False, **kw))
                exec(source, ns)
                with backend.backend_environment(selected):
                    frames.append(ns['run_baselines_cv'](X, y, splits))
            self.assert_csv_contract(*frames)
            self.assertEqual(frames[0].split_fingerprint.tolist(), frames[1].split_fingerprint.tolist())


if __name__ == '__main__':
    unittest.main()
