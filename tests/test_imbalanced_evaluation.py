"""Tests for imbalanced evaluation; synthetic scores only."""
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from imbalanced_evaluation import (
    evaluate_binary, evaluate_with_calibration, tune_f1_threshold
)


class ImbalancedEvaluationTests(unittest.TestCase):
    def test_pr_auc_prevalence_baseline(self):
        y = [0, 0, 0, 1]
        scores = [0.1, 0.2, 0.3, 0.9]
        metrics = evaluate_binary(y, scores)
        self.assertAlmostEqual(metrics["pr_auc"], 1)
        self.assertAlmostEqual(metrics["pr_baseline"], .25)
        self.assertAlmostEqual(metrics["majority_accuracy"], .75)
        self.assertAlmostEqual(metrics["majority_f1"], 0)
        self.assertAlmostEqual(metrics["auc"], 1)

    def test_tune_uses_only_calibration_scores(self):
        y_cal = [0, 0, 0, 1]
        scores_cal = [.02, .03, .04, .2]
        threshold = tune_f1_threshold(y_cal, scores_cal)
        self.assertAlmostEqual(threshold, .2)
        a = evaluate_with_calibration([0, 1], [.25, .15], y_cal, scores_cal)
        b = evaluate_with_calibration([1, 0], [.25, .15], y_cal, scores_cal)
        self.assertEqual(a["threshold_tuned"], b["threshold_tuned"])
        self.assertNotEqual(a["f1_tuned"], b["f1_tuned"])

    def test_ties_and_degenerate_calibration(self):
        self.assertAlmostEqual(tune_f1_threshold([0, 1], [.4, .4]), .4)
        with self.assertRaises(ValueError):
            tune_f1_threshold([0, 0], [.1, .2])
        result = evaluate_binary([0, 0], [.1, .2])
        self.assertTrue(np.isnan(result["auc"]))
        self.assertTrue(np.isnan(result["pr_auc"]))


if __name__ == "__main__":
    unittest.main()
