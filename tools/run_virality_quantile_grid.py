#!/usr/bin/env python3
"""Run the same 4 viral quantiles on EVONS and FakeNewsNet with cached embeddings.

CPU/GPU training and original data are required. The script never re-embeds.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q_VALUES = (0.5, 0.75, 0.9, 0.95)


def validate_quantiles(values):
    quantiles = [float(v) for v in values.split(",")]
    if not quantiles or any(not 0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must all lie strictly within (0,1)")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("Duplicate quantiles are not allowed")
    return quantiles


def run(dataset="both", quantiles=Q_VALUES):
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
            previous = os.environ.get("VIRALITY_QUANTILE")
            os.environ["VIRALITY_QUANTILE"] = str(q)
            try:
                NotebookClient(
                    nb, timeout=7200, kernel_name="python3",
                    resources={"metadata": {"path": str(ROOT)}}
                ).execute()
            finally:
                if previous is None:
                    os.environ.pop("VIRALITY_QUANTILE", None)
                else:
                    os.environ["VIRALITY_QUANTILE"] = previous
            # Metrics are already saved by the notebook to CSV.
            # Do not persist executed cells, outputs, or raw tweet text.
            target = ROOT / "colab_no_embeddings/outputs" / f"fakenewsnet_virality_q{label}"
            if not target.is_dir():
                raise RuntimeError(f"Missing expected output directory: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("evons", "fakenewsnet", "both"), default="both")
    parser.add_argument("--quantiles", default="0.5,0.75,0.9,0.95")
    args = parser.parse_args()
    run(args.dataset, validate_quantiles(args.quantiles))


if __name__ == "__main__":
    main()
