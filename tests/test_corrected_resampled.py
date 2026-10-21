"""Corrected repeated-holdout test and quantile convention regression tests."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from corrected_resampled_tests import corrected_pair, analyze, holm
from fakenewsnet_virality_sensitivity import summarize_thresholds, early_signal_rows


class CorrectedTests(unittest.TestCase):
    def test_nadeau_bengio_factor(self):
        a = [.8, .9, .7, .85, .8, .75, .84, .87, .81, .82]
        b = [.7] * 10
        result = corrected_pair(a, b, train_fraction=.8)
        self.assertAlmostEqual(result["correction_factor"], .35)
        self.assertEqual(result["df"], 9)
        self.assertTrue(result["corrected_se"] > 0)
        self.assertTrue(0 <= result["p_value_corrected"] <= 1)

    def test_holm_and_pairing(self):
        self.assertEqual(holm([.01, .03, .05]), [.03, .06, .06])
        rows = []
        for fold, (a, b) in enumerate(zip([.3, .5, .7], [.2, .25, .3]), 1):
            for model, v in [("A", a), ("B", b)]:
                rows.append(dict(dataset="evons",task="virality",metric="f1",
                                 fold=str(fold),model=model,value=str(v)))
        result = analyze(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["n_paired_splits"], 3)

    def test_no_pairs_or_invalid_data(self):
        with self.assertRaises(ValueError):
            corrected_pair([.1], [.2])
        with self.assertRaises(ValueError):
            corrected_pair([.1, .2, .3], [.2, .3, .4], train_fraction=1)

    def test_strict_virality_cutoff_with_ties(self):
        vals = [0, 0, 0, 1]
        report = summarize_thresholds(vals, [.5])
        self.assertEqual(report[0]["n_positive"], 1)
        out = early_signal_rows(vals, {1: [0, 0, 0, 1]}, [.5])
        self.assertAlmostEqual(out[0]["auc_prefix_for_label"], 1)


if __name__ == "__main__":
    unittest.main()
