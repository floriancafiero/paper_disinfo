#!/usr/bin/env python3
"""Run the same 4 viral quantiles on EVONS and FakeNewsNet with cached embeddings.

CPU/GPU training and original data are required. The script never re-embeds.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from fakenewsnet_prefix import DEFAULT_WINDOWS, validate_windows


ROOT = Path(__file__).resolve().parents[1]
Q_VALUES = (0.5, 0.75, 0.9, 0.95)


def validate_quantiles(values):
    quantiles = [float(v) for v in values.split(",")]
    if not quantiles or any(not 0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must all lie strictly within (0,1)")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("Duplicate quantiles are not allowed")
    return quantiles


def run(dataset="both", quantiles=Q_VALUES, fnn_windows=DEFAULT_WINDOWS):
    validate_windows(fnn_windows)
    for q in quantiles:
        env = dict(os.environ, VIRALITY_QUANTILE=str(q))
        label = str(round(q * 100))
        if dataset in ("evons", "both"):
            print(f"Running EVONS q={q}", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "colab_no_embeddings/evons_virality_prediction.py")],
                cwd=ROOT, env=env, check=True
            )
        if dataset in ("fakenewsnet", "both"):
            print(f"Running FakeNewsNet q={q}", flush=True)
            # nbclient launches a notebook kernel; preserve its environment
            # for the VIRALITY_QUANTILE variable and work from repository root.
            import nbformat
            from nbclient import NotebookClient
            source = ROOT / "colab_no_embeddings/fakenewsnet_virality_prediction.ipynb"
            nb = nbformat.read(source, as_version=4)
            settings = {"VIRALITY_QUANTILE": str(q), "FNN_WINDOWS": fnn_windows}
            previous = {key: os.environ.get(key) for key in settings}
            os.environ.update(settings)
            try:
                NotebookClient(
                    nb, timeout=7200, kernel_name="python3",
                    resources={"metadata": {"path": str(ROOT)}}
                ).execute()
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            # Metrics are already saved by the notebook to CSV.
            # Do not persist executed cells, outputs, or raw tweet text.
            target = ROOT / "colab_no_embeddings/outputs" / f"fakenewsnet_virality_prefix_q{label}"
            for name in ("fakenewsnet_virality_merged_cv_results.csv", "fakenewsnet_virality_merged_summary.csv"):
                if not (target / name).is_file():
                    raise RuntimeError(f"Missing expected output: {target / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("evons", "fakenewsnet", "both"), default="both")
    parser.add_argument("--quantiles", default="0.5,0.75,0.9,0.95")
    parser.add_argument("--fnn-windows", default=DEFAULT_WINDOWS,
                        help="FakeNewsNet text windows selected from 1,3,5,10,full; cohort always >=11 tweets")
    args = parser.parse_args()
    run(args.dataset, validate_quantiles(args.quantiles), args.fnn_windows)


if __name__ == "__main__":
    main()
