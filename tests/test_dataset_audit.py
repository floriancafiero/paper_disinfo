"""Dataset-audit regression tests. Run: python -m unittest discover -s tests -v"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dataset_audit import quantile, thresholds, audit_evons, audit_fakenewsnet


class DatasetAuditTests(unittest.TestCase):
    def test_quantile_ties(self):
        self.assertEqual(quantile([1, 2, 3, 4], 0.5), 2.5)
        rows = thresholds([0, 0, 0, 1], (0.5, 0.75, 0.95))
        self.assertEqual(rows[0]["positive_rate"], 0.25)
        self.assertEqual(rows[0]["n_tied_at_threshold"], 3)
        self.assertEqual(rows[-1]["n_positive"], 1)

    def test_evons(self):
        with tempfile.TemporaryDirectory() as d:
            csvfile = Path(d) / "evons.csv"
            with csvfile.open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=["is_fake", "fb_engagements"])
                writer.writeheader()
                writer.writerows([
                    {"is_fake": 1, "fb_engagements": 0},
                    {"is_fake": 0, "fb_engagements": 0},
                    {"is_fake": 1, "fb_engagements": 4},
                    {"is_fake": 0, "fb_engagements": ""},
                ])
            report = audit_evons(csvfile, (0.5,))
            self.assertEqual(report["n_rows"], 4)
            self.assertEqual(report["n_missing_engagement"], 1)
            self.assertEqual(report["thresholds"][0]["n_positive"], 1)

    def test_propagation_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake, real = d / "fake.jsonl", d / "real.jsonl"
            fake.write_text("\n".join(map(json.dumps, [
                [{"favorite_count": 2}, {"favorite_count": 3}],
                [{"text": "missing engagement"}],
            ])) + "\n")
            real.write_text(json.dumps([
                {"favorite_count": 10}, {"favorite_count": 20}, {"favorite_count": 30}
            ]) + "\n")
            report = audit_fakenewsnet(fake, real, (0.5,), (1, 2, 3))
            self.assertEqual(report["veracity_counts"], {"fake": 2, "real": 1})
            coverage = report["propagation_coverage_within_processed_jsonl"]
            self.assertEqual(coverage["n_with_future_tweets_after_k"], {"1": 2, "2": 1, "3": 0})
            self.assertEqual(coverage["n_tweets_missing_favorite_count_treated_as_zero"], 1)
            self.assertEqual(report["thresholds"]["pooled_main_task"][0]["threshold"], 5.0)

    def test_original_directory_counts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for label in ("fake", "real"):
                (d / label / "a" / "tweets").mkdir(parents=True)
                (d / label / "a" / "tweets" / "x.json").write_text("{}")
                (d / label / "b").mkdir()
            fake, real = d / "fake.jsonl", d / "real.jsonl"
            fake.write_text(json.dumps([{"favorite_count": 1}]) + "\n")
            real.write_text(json.dumps([{"favorite_count": 2}]) + "\n")
            report = audit_fakenewsnet(fake, real, (0.5,), (1,), raw_root=d)
            self.assertEqual(report["optional_original_story_directory_counts"]["fake"]["n_story_directories"], 2)
            self.assertEqual(report["optional_original_story_directory_counts"]["real"]["n_directories_with_tweet_json"], 1)


if __name__ == "__main__":
    unittest.main()
