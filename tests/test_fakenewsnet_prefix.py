"""Behavioral tests for text-only, matched-cohort FakeNewsNet prefixes."""
import copy
import csv
import io
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fakenewsnet_prefix import prepare_features, make_splits, validate_windows, split_fingerprint
from imbalanced_evaluation import evaluate_with_calibration


def fixture(n=20):
    paths, embeddings = {}, {}
    for category in ("fake", "real"):
        paths[category], embeddings[category] = [], []
        for i in range(n):
            # One excluded story per category; all others have >=11 tweets.
            length = 3 if i == 0 else 11 + i % 3
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            paths[category].append([
                {"created_at": (start + timedelta(seconds=j)).strftime("%a %b %d %H:%M:%S %z %Y"),
                 "id_str": f"{category}-{i}-{j}", "favorite_count": i}
                for j in range(length)
            ])
            embeddings[category].append([
                np.array([j, i + int(category == 'fake')], dtype=np.float32)
                for j in range(length)
            ])
    return paths, embeddings


class PrefixTests(unittest.TestCase):
    def test_prefix_ignores_all_later_text(self):
        paths, embeddings = fixture()
        for k in (1, 3, 5, 10):
            windows = [str(k), 'full']
            original, y, metadata = prepare_features(paths, embeddings, windows, 'virality')
            changed = copy.deepcopy(embeddings)
            for category in changed:
                for seq in changed[category]:
                    for vector in seq[k:]:
                        vector[:] += 1000
            modified, y2, metadata2 = prepare_features(paths, changed, windows, 'virality')
            np.testing.assert_array_equal(original[str(k)], modified[str(k)])
            self.assertFalse(np.array_equal(original['full'], modified['full']))
            np.testing.assert_array_equal(y, y2)
            self.assertEqual(metadata, metadata2)
            np.testing.assert_allclose(original[str(k)][:, 0], (k - 1) / 2)

    def test_cohort_total_target_and_encoders_match(self):
        paths, embeddings = fixture()
        features, y, meta = prepare_features(paths, embeddings, ['1', '10', 'full'], 'virality', .75)
        self.assertTrue(all(len(x) == 38 for x in features.values()))
        self.assertEqual(meta['n_excluded_fake'], 1)
        self.assertEqual(meta['n_cohort_real'], 19)
        totals = [sum(p['favorite_count'] for p in story) for group in paths.values() for story in group if len(story) >= 11]
        threshold = np.quantile(totals, .75)
        self.assertEqual(meta['virality_threshold'], threshold)
        np.testing.assert_array_equal(y, np.asarray(totals) > threshold)
        # A second encoder may have a different feature dimension, not a different cohort.
        second = {c: [np.pad(np.array(s), ((0, 0), (0, 1))) for s in seqs] for c, seqs in embeddings.items()}
        _, y2, meta2 = prepare_features(paths, second, ['3'], 'virality', .75)
        np.testing.assert_array_equal(y, y2)
        self.assertEqual(meta, meta2)

    def test_validation_rejects_corrupt_inputs(self):
        paths, embeddings = fixture(2)
        cases = []
        e = copy.deepcopy(embeddings); e['fake'].pop(); cases.append((paths, e, 'story count mismatch'))
        e = copy.deepcopy(embeddings); e['fake'][1].pop(); cases.append((paths, e, 'tweet count mismatch'))
        e = copy.deepcopy(embeddings); e['fake'][1][0][0] = np.nan; cases.append((paths, e, 'nonfinite'))
        e = copy.deepcopy(embeddings); e['fake'][1][0] = np.zeros(3); cases.append((paths, e, 'equal-sized'))
        e = {c: np.array([np.mean(s, axis=0) for s in seqs]) for c, seqs in embeddings.items()}; cases.append((paths, e, 'per-tweet'))
        p = copy.deepcopy(paths); p['real'][1].reverse(); cases.append((p, embeddings, 'chronologically'))
        p = copy.deepcopy(paths); del p['fake'][1][0]['created_at']; cases.append((p, embeddings, 'created_at'))
        p = copy.deepcopy(paths); p['fake'][1][0]['favorite_count'] = float('inf'); cases.append((p, embeddings, 'favorite_count'))
        for p, e, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prepare_features(p, e, ['1'], 'virality')

    def test_window_validation_and_small_cohorts(self):
        self.assertEqual(validate_windows('1,3,5,10,full'), ['1', '3', '5', '10', 'full'])
        for value in ('', '0', '11', '1,1', 'all', '3,'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_windows(value)
        for y in ([0] * 20, [0] * 19 + [1]):
            with self.assertRaises(ValueError):
                make_splits(y)
        paths, embeddings = fixture(1)
        with self.assertRaisesRegex(ValueError, 'No FakeNewsNet stories'):
            prepare_features(paths, embeddings, ['1'], 'disinformation')

    def test_lightweight_training_and_csv_roundtrip(self):
        paths, embeddings = fixture(40)
        features, y, meta = prepare_features(paths, embeddings, ['1', '3', 'full'], 'disinformation')
        splits = make_splits(y)
        again = make_splits(y)
        records = []
        for first, second in zip(splits, again):
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)
        for window, X in features.items():
            for fold, (fit, cal, test) in enumerate(splits, 1):
                self.assertFalse(set(fit) & set(cal) or set(fit) & set(test) or set(cal) & set(test))
                self.assertEqual(len(fit) + len(cal) + len(test), len(y))
                model = LogisticRegression().fit(X[fit], y[fit])
                result = evaluate_with_calibration(y[test], model.predict_proba(X[test])[:, 1], y[cal], model.predict_proba(X[cal])[:, 1])
                result.update(meta, fold=fold, model='logreg', observation_window=window,
                              split_fingerprint=split_fingerprint(fit, cal, test))
                records.append(result)
        csvfile = io.StringIO()
        writer = csv.DictWriter(csvfile, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records); csvfile.seek(0)
        rows = list(csv.DictReader(csvfile))
        self.assertEqual(len(rows), 30)
        self.assertEqual({r['observation_window'] for r in rows}, {'1', '3', 'full'})
        for fold in range(1, 11):
            self.assertEqual(len({r['split_fingerprint'] for r in rows if r['fold'] == str(fold)}), 1)
        self.assertEqual(rows[0]['n_positive_cohort'], str(int(y.sum())))
        self.assertNotEqual(rows[0]['n_positive'], rows[0]['n_positive_cohort'])

    def test_notebooks_preserve_window_grouping_and_remove_stale_outputs(self):
        for name in ('fakenewsnet_virality_prediction', 'fakenewsnet_disinformation_detection'):
            nb = json.loads((ROOT / 'colab_no_embeddings' / f'{name}.ipynb').read_text())
            source = '\n'.join(''.join(c['source']) for c in nb['cells'])
            self.assertIn("groupby(['model', 'observation_window'])", source)
            self.assertNotIn('to_tensor_embeddings', source)
            for c in nb['cells']:
                if c['cell_type'] == 'code':
                    self.assertEqual(c['outputs'], [])
                    self.assertIsNone(c['execution_count'])


if __name__ == '__main__':
    unittest.main()
