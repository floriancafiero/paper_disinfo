"""Safe packaging tests: do not inadvertently share arbitrary local datasets."""
import importlib.util
import json
import tempfile
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("colleague_revision_run", ROOT / "tools/colleague_revision_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ColleagueHandoffTests(unittest.TestCase):
    def test_archive_only_approved_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            out = tmp / "revision_results"
            out.mkdir()
            (out / "dataset_audit.json").write_text('{"counts": 2}')
            (out / "preflight_manifest.json").write_text('{"data_inventory": {}}')
            (out / "raw_articles.csv").write_text("sensitive,content")
            (out / "future_fake.csv").write_text("quantile,auc\\n.5,.6\\n")
            output_folder = tmp / "colab_no_embeddings/outputs/evons_virality_q95"
            output_folder.mkdir(parents=True)
            (output_folder / "evons_virality_merged_cv_results.csv").write_text("fold,f1\\n1,.5\\n")
            (output_folder / "raw_tweets.csv").write_text("sensitive")
            prefix_folder = tmp / "colab_no_embeddings/outputs/fakenewsnet_virality_prefix_q50"
            prefix_folder.mkdir()
            (prefix_folder / "fakenewsnet_virality_merged_cv_results.csv").write_text("fold,observation_window,f1\n1,k1,.5\n")
            legacy_folder = tmp / "colab_no_embeddings/outputs/fakenewsnet_virality_q50"
            legacy_folder.mkdir()
            (legacy_folder / "fakenewsnet_virality_merged_cv_results.csv").write_text("historical")
            with patch.object(runner, "ROOT", tmp), patch.object(runner, "OUT", out):
                runner.make_archive()
            archive = out / "SHARE_AGGREGATES_ONLY.zip"
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
            self.assertIn("revision_results/dataset_audit.json", names)
            self.assertIn("revision_results/future_fake.csv", names)
            self.assertIn("colab_no_embeddings/outputs/evons_virality_q95/evons_virality_merged_cv_results.csv", names)
            self.assertNotIn("revision_results/raw_articles.csv", names)
            self.assertFalse(any("raw_tweets" in n for n in names))
            self.assertIn("colab_no_embeddings/outputs/fakenewsnet_virality_prefix_q50/fakenewsnet_virality_merged_cv_results.csv", names)
            self.assertFalse(any("/fakenewsnet_virality_q50/" in n for n in names))

    def test_check_without_any_files(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            out = tmp / "revision_results"
            with patch.object(runner, "ROOT", tmp), patch.object(runner, "OUT", out), patch.object(
                runner, "REQUIRED", {"fake": [tmp / "does_not_exist.csv"]}
            ):
                self.assertFalse(runner.check())
            self.assertTrue((out / "preflight_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
