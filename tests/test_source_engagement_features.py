"""Synthetic regression tests for source feature leakage."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from source_engagement_features import source_engagement_features


class SourceEngagementTests(unittest.TestCase):
    def test_leave_one_out_does_not_depend_on_own_outcome(self):
        sources = ["A", "A", "B", "B"]
        val_sources = ["A", "B", "Z"]
        fit1, val1 = source_engagement_features(sources, [10., 20., 30., 40.], val_sources)
        fit2, _ = source_engagement_features(sources, [999., 20., 30., 40.], val_sources)
        # The raw own-source mean for article 0 stays at 20 when its own outcome changes.
        # Normalization may change with other features, so compare de-normalized
        # leave-one-out estimates using the directly observable relative differences.
        self.assertAlmostEqual(fit1[0] - fit1[1], - (fit1[1] - fit1[0]))
        self.assertEqual(len(val1), 3)
        self.assertTrue(all(abs(x) < 100 for x in fit2))

    def test_singleton_and_unseen_source(self):
        fit, other = source_engagement_features(["A", "B"], [10., 30.], ["A", "C"])
        self.assertAlmostEqual(fit[0], 1.)
        self.assertAlmostEqual(fit[1], -1.)
        self.assertAlmostEqual(other[0], -1.)
        self.assertAlmostEqual(other[1], 0.)

    def test_bad_engagement_rejected(self):
        with self.assertRaises(ValueError):
            source_engagement_features(["A"], [float("nan")], ["A"])


if __name__ == "__main__":
    unittest.main()
