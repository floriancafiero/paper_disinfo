import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class ToolScriptsSmokeTests(unittest.TestCase):
    def run_cmd(self, args):
        subprocess.run(args, cwd=REPO_ROOT, check=True)

    def test_statistical_evaluation_long_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "metrics.csv"
            out = tmpdir / "stats.csv"
            rows = [
                ["dataset", "task", "metric", "fold", "model", "value"],
                ["evons", "disinfo", "acc", "0", "A", "0.80"],
                ["evons", "disinfo", "acc", "0", "B", "0.70"],
                ["evons", "disinfo", "acc", "1", "A", "0.82"],
                ["evons", "disinfo", "acc", "1", "B", "0.72"],
                ["evons", "disinfo", "acc", "2", "A", "0.78"],
                ["evons", "disinfo", "acc", "2", "B", "0.69"],
            ]
            with inp.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

            self.run_cmd([
                "python",
                "tools/statistical_evaluation.py",
                "--input",
                str(inp),
                "--output",
                str(out),
                "--n-boot",
                "200",
            ])

            self.assertTrue(out.exists())
            with out.open(newline="", encoding="utf-8") as f:
                out_rows = list(csv.DictReader(f))
            self.assertEqual(len(out_rows), 1)
            self.assertEqual(out_rows[0]["model_a"], "A")

    def test_evons_source_confounding_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "evons.csv"
            metrics_out = tmpdir / "metrics.csv"
            report_out = tmpdir / "report.json"
            folds_out = tmpdir / "folds.csv"

            rows = [
                ["id", "media_source", "label"],
                ["1", "src_a", "fake"],
                ["2", "src_a", "fake"],
                ["3", "src_b", "real"],
                ["4", "src_b", "real"],
                ["5", "src_c", "fake"],
                ["6", "src_c", "real"],
            ]
            with inp.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

            self.run_cmd([
                "python",
                "tools/evons_source_confounding_audit.py",
                "--input",
                str(inp),
                "--folds",
                "3",
                "--metrics-out",
                str(metrics_out),
                "--report-out",
                str(report_out),
                "--export-group-folds",
                str(folds_out),
                "--id-col",
                "id",
            ])

            self.assertTrue(metrics_out.exists())
            self.assertTrue(report_out.exists())
            self.assertTrue(folds_out.exists())
            with report_out.open(encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["n_rows"], 6)

    def test_fakenewsnet_virality_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "props.jsonl"
            thresholds_out = tmpdir / "thresholds.csv"
            early_out = tmpdir / "early.csv"
            summary_out = tmpdir / "summary.json"

            propagations = [
                [{"favorite_count": 1}, {"favorite_count": 2}, {"favorite_count": 3}],
                [{"favorite_count": 0}, {"favorite_count": 1}],
                [{"favorite_count": 4}, {"favorite_count": 4}, {"favorite_count": 2}],
                [{"favorite_count": 2}],
            ]
            with inp.open("w", encoding="utf-8") as f:
                for seq in propagations:
                    f.write(json.dumps(seq) + "\n")

            self.run_cmd([
                "python",
                "tools/fakenewsnet_virality_sensitivity.py",
                "--input-jsonl",
                str(inp),
                "--quantiles",
                "0.5,0.9",
                "--k-prefix",
                "1,2",
                "--thresholds-out",
                str(thresholds_out),
                "--early-out",
                str(early_out),
                "--summary-out",
                str(summary_out),
            ])

            self.assertTrue(thresholds_out.exists())
            self.assertTrue(early_out.exists())
            self.assertTrue(summary_out.exists())
            with summary_out.open(encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["n_propagations"], 4)


if __name__ == "__main__":
    unittest.main()
