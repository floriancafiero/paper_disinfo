"""Small real notebook-cell integration, optional when PyTorch is installed."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fakenewsnet_prefix import DEFAULT_WINDOWS, make_splits, prepare_features, split_fingerprint, validate_windows
from imbalanced_evaluation import evaluate_with_calibration
from test_fakenewsnet_prefix import fixture


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'pandas', 'tabulate')),
                     'Notebook integration requires optional torch, pandas and tabulate')
class NotebookIntegrationTests(unittest.TestCase):
    def test_both_notebook_load_train_and_export(self):
        import pandas as pd
        import torch
        from torch import nn
        from torch.utils.data import Dataset, DataLoader
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for task in ('virality', 'disinformation'):
                with self.subTest(task=task), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    data_dir = root / 'FakeNewsNet/data'
                    (data_dir / 'fake_news_net').mkdir(parents=True)
                    paths, embeddings = fixture(40)
                    for category in paths:
                        (data_dir / f'ordered_{category}_propagation_paths.jsonl').write_text(
                            '\n'.join(json.dumps(posts) for posts in paths[category]) + '\n')
                        for mode, suffix in (('bert', ''), ('mistral', '_mistral')):
                            nested = [[torch.tensor(v if mode == 'bert' else np.pad(v, (0, 1))) for v in story]
                                      for story in embeddings[category]]
                            torch.save(nested, data_dir / 'fake_news_net' / f'ordered_{category}_propagation_paths_emb{suffix}.pt')
                    name = f'fakenewsnet_{task}_{"prediction" if task == "virality" else "detection"}'
                    nb = json.loads((ROOT / 'colab_no_embeddings' / f'{name}.ipynb').read_text())
                    ns = dict(Path=Path, os=os, json=json, np=np, pd=pd, torch=torch, nn=nn,
                              Dataset=Dataset, DataLoader=DataLoader, ROOT=root, DATA_DIR=data_dir,
                              DEFAULT_WINDOWS=DEFAULT_WINDOWS, validate_windows=validate_windows,
                              prepare_features=prepare_features, split_fingerprint=split_fingerprint,
                              make_splits=lambda y: make_splits(y, n_splits=2),
                              evaluate_with_calibration=evaluate_with_calibration, HAS_XGB=False,
                              LogisticRegression=lambda **kw: LogisticRegression(max_iter=1000, n_jobs=1, random_state=42),
                              RandomForestClassifier=lambda **kw: RandomForestClassifier(n_estimators=2, n_jobs=1, random_state=42))
                    for i in (2, 3):
                        exec(''.join(nb['cells'][i]['source']), ns)
                    original_mlp = ns['run_mlp_cv']
                    ns['run_mlp_cv'] = lambda X, y, splits: original_mlp(X, y, splits, epochs=1)
                    with patch.dict(os.environ, {'FNN_WINDOWS': '1,full', 'VIRALITY_QUANTILE': '0.5'}):
                        for i in (4, 5, 6, 7):
                            exec(''.join(nb['cells'][i]['source']), ns)
                    folder = root / 'colab_no_embeddings/outputs' / (
                        'fakenewsnet_virality_prefix_q50' if task == 'virality' else 'fakenewsnet_disinformation_prefix')
                    results = pd.read_csv(folder / f'fakenewsnet_{task}_merged_cv_results.csv')
                    summary = pd.read_csv(folder / f'fakenewsnet_{task}_merged_summary.csv')
                    self.assertEqual(len(results), 24)  # 2 encoders x 2 windows x 3 models x 2 folds
                    self.assertEqual(len(summary), 12)
                    self.assertEqual(set(results.observation_window), {'k1', 'full'})
                    self.assertTrue((results.groupby('fold').split_fingerprint.nunique() == 1).all())
                    self.assertEqual(results.cohort_fingerprint.nunique(), 1)
                    self.assertEqual(results.target_type.unique().tolist(),
                                     ['total_engagement' if task == 'virality' else 'veracity'])
                    self.assertTrue((results.n_cohort == 78).all())
                    self.assertIn('full', (folder / f'fakenewsnet_{task}_publication_summary.md').read_text())
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
