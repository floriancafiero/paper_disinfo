"""Regression tests for prospective, prefix-excluded virality target."""
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fakenewsnet_future_virality import prospective_rows, auc, average_precision


class FutureViralityTests(unittest.TestCase):
    def test_rank_metrics_and_ties(self):
        self.assertAlmostEqual(auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.)
        self.assertAlmostEqual(auc([1, 1], [0, 1]), .5)
        self.assertAlmostEqual(average_precision([1, 1], [0, 1]), .5)
        self.assertAlmostEqual(average_precision([1, 2, 3, 4], [0, 0, 1, 1]), 1.)
        self.assertTrue(str(auc([1, 2], [0, 0])) == "nan")

    def test_future_excludes_prefix_and_fixed_cohort(self):
        # k=1 prefix scores [1,2,3,4]; post-prefix engagement [10,20,30,40].
        seqs = [[1, 10, 0], [2, 20, 0], [3, 30, 0], [4, 40, 0], [999]]
        rows, coverage = prospective_rows(seqs, ks=(1, 2), qs=(.5,), cohort="fixed")
        self.assertEqual([r["n_sequences"] for r in rows], [4, 4])
        self.assertEqual(coverage["n_with_future_posts_after_k"], {"1": 4, "2": 4})
        self.assertEqual(rows[0]["threshold_future_likes"], 25)
        self.assertEqual(rows[0]["n_positive"], 2)
        self.assertAlmostEqual(rows[0]["auc_prefix_for_future_label"], 1.)
        self.assertEqual(rows[1]["threshold_future_likes"], 0)
        self.assertEqual(rows[1]["n_positive"], 0)
        self.assertNotEqual(rows[0]["threshold_future_likes"], rows[1]["threshold_future_likes"])

    def test_cohort_change_and_missing_future(self):
        seqs = [[1, 100], [2, 200], [3, 5, 1], [4, 10, 2], [9]]
        fixed, _ = prospective_rows(seqs, (1, 2), (.5,), cohort="fixed")
        eligible, _ = prospective_rows(seqs, (1, 2), (.5,), cohort="eligible")
        self.assertEqual([r["n_sequences"] for r in fixed], [2, 2])
        self.assertEqual([r["n_sequences"] for r in eligible], [4, 2])

    def test_cli_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            inp = p / "propagation.jsonl"
            out, summary = p / "future.csv", p / "future.json"
            seqs = [[{"favorite_count": x}, {"favorite_count": y}]
                    for x, y in [(1, 10), (2, 20), (3, 30), (4, 40)]]
            inp.write_text("".join(json.dumps(s) + "\n" for s in seqs))
            subprocess.run([sys.executable, str(ROOT / "tools/fakenewsnet_future_virality.py"),
                            "--input-jsonl", str(inp), "--k-prefix", "1", "--quantiles", ".5",
                            "--out", str(out), "--summary-out", str(summary)], check=True)
            with out.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["threshold_future_likes"]), 25.)
            self.assertEqual(float(rows[0]["auc_prefix_for_future_label"]), 1.)
            self.assertEqual(json.loads(summary.read_text())["coverage"]["n_fixed_cohort"], 4)


if __name__ == "__main__":
    unittest.main()
