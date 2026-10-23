"""Smoke-test inference assumptions for revised overlapping 60/20/20 splits."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from revised_cv_inference import difference_stats, adjust_holm, load_models, analyze


class RevisedInferenceTests(unittest.TestCase):
    def test_variance_inflation_and_identical_folds(self):
        a = {str(i): .70 + .02 * (i % 3) for i in range(1, 11)}
        b = {str(i): .65 + .01 * (i % 4) for i in range(1, 11)}
        out = difference_stats(a, b)
        self.assertEqual(out["n_paired_splits"], 10)
        self.assertAlmostEqual(out["correction_factor"], 1 / 10 + 1 / 3)
        self.assertGreater(out["corrected_se"], 0)
        with self.assertRaises(ValueError):
            difference_stats(a, {**b, "11": .5})

    def test_holm_within_family(self):
        rows = [{"corrected_p": .01}, {"corrected_p": .04}, {"corrected_p": .03}]
        adjust_holm(rows)
        self.assertEqual([round(r["holm_p"], 2) for r in rows], [.03, .06, .06])
        self.assertEqual([r["significant_005"] for r in rows], [True, False, False])

    def test_checks_split_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "dummy.csv"
            def write(change=False):
                with file.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["model", "fold", "observation_window", "f1_tuned", "cohort_fingerprint", "split_fingerprint"])
                    writer.writeheader()
                    for model in ("A", "B"):
                        for window in ("k1", "full"):
                            for i in range(1, 11):
                                key = f"sha-{i}"
                                if change and model == "B" and window == "k1" and i == 1:
                                    key = "wrong-sha"
                                writer.writerow(dict(model=model, fold=i, observation_window=window,
                                                     f1_tuned=.75, cohort_fingerprint="same",
                                                     split_fingerprint=key))
            write()
            grouped, metadata = load_models(file)
            self.assertEqual(len(grouped), 2)
            self.assertEqual(metadata["cohort_fingerprint"], "same")
            write(change=True)
            with self.assertRaisesRegex(ValueError, "Unpaired split"):
                load_models(file)

    def test_repo_files_are_consistent(self):
        # This is an actual audit of the tracked revised results, not a synthetic check.
        models, windows, integrity = analyze()
        self.assertEqual(len(integrity), 10)
        self.assertTrue(len(models) > 100)
        self.assertEqual({x["dataset"] for x in windows}, {"fakenewsnet"})
        self.assertTrue(all(x["n_paired_splits"] == 10 for x in windows))


if __name__ == "__main__":
    unittest.main()
