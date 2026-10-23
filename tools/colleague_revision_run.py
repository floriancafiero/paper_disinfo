#!/usr/bin/env python3
"""PLOS ONE revision handoff runner. No corpus/embedding upload or API calls.

Usage (from project root):
  python tools/colleague_revision_run.py --stage check
  python tools/colleague_revision_run.py --stage audit
  python tools/colleague_revision_run.py --stage train
  python tools/colleague_revision_run.py --stage all

The shareable ZIP only contains aggregate CSV/JSON and a manifest, never raw
JSONL, text, embeddings, intermediate predictions or executed notebooks.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fakenewsnet_prefix import DEFAULT_WINDOWS, validate_windows
from baseline_backend import resolve_backend, backend_environment, require_gpu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_results"
EVONS = ROOT / "evons/data"
FNN = ROOT / "FakeNewsNet/data"
QUANTILES = ("0.5", "0.75", "0.9", "0.95")
REQUIRED = {
    "EVONS": [EVONS / p for p in (
        "evons.csv", "title_embeddings.pt", "desc_embeddings.pt",
        "title_embeddings_mistral.pt", "desc_embeddings_mistral.pt"
    )],
    "FakeNewsNet": [FNN / p for p in (
        "ordered_fake_propagation_paths.jsonl",
        "ordered_real_propagation_paths.jsonl",
        "fake_news_net/ordered_fake_propagation_paths_emb.pt",
        "fake_news_net/ordered_real_propagation_paths_emb.pt",
        "fake_news_net/ordered_fake_propagation_paths_emb_mistral.pt",
        "fake_news_net/ordered_real_propagation_paths_emb_mistral.pt",
    )],
}


def run(command, env=None):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT, env=env)


def check():
    manifest = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0], "platform": platform.platform(),
                "data_inventory": {}}
    complete = True
    for name, paths in REQUIRED.items():
        inventory = {}
        for p in paths:
            exists = p.is_file()
            inventory[str(p.relative_to(ROOT))] = {
                "present": exists, "size_bytes": p.stat().st_size if exists else None
            }
            print(("OK      " if exists else "MISSING ") + str(p.relative_to(ROOT)))
            complete &= exists
        manifest["data_inventory"][name] = inventory
    OUT.mkdir(exist_ok=True)
    (OUT / "preflight_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return complete


def audit():
    OUT.mkdir(exist_ok=True)
    run([sys.executable, "tools/dataset_audit.py",
         "--evons", REQUIRED["EVONS"][0],
         "--fake-jsonl", REQUIRED["FakeNewsNet"][0],
         "--real-jsonl", REQUIRED["FakeNewsNet"][1],
         "--out", OUT / "dataset_audit.json"])
    for label, path in [("fake", REQUIRED["FakeNewsNet"][0]),
                        ("real", REQUIRED["FakeNewsNet"][1])]:
        run([sys.executable, "tools/fakenewsnet_future_virality.py",
             "--input-jsonl", path, "--cohort", "fixed",
             "--out", OUT / f"future_{label}.csv",
             "--summary-out", OUT / f"future_{label}_summary.json"])
        run([sys.executable, "tools/fakenewsnet_future_virality.py",
             "--input-jsonl", path, "--cohort", "eligible",
             "--out", OUT / f"future_{label}_eligible.csv",
             "--summary-out", OUT / f"future_{label}_eligible_summary.json"])
        run([sys.executable, "tools/fakenewsnet_virality_sensitivity.py",
             "--input-jsonl", path,
             "--thresholds-out", OUT / f"thresholds_{label}.csv",
             "--early-out", OUT / f"legacy_total_{label}.csv",
             "--summary-out", OUT / f"legacy_total_{label}_summary.json"])


def execute_notebook(path, label, fnn_windows=DEFAULT_WINDOWS, baseline_backend=None, kernel_name="python3"):
    from nbclient import NotebookClient
    import nbformat
    nb = nbformat.read(str(ROOT / path), as_version=4)
    print(f"Starting {label}; this can take hours", flush=True)
    # Avoid saving executed notebooks; they can contain raw text in outputs.
    validate_windows(fnn_windows)
    previous = os.environ.get("FNN_WINDOWS")
    os.environ["FNN_WINDOWS"] = fnn_windows
    try:
        with backend_environment(resolve_backend(baseline_backend)):
            NotebookClient(nb, timeout=7200, kernel_name=kernel_name,
                           resources={"metadata": {"path": str(ROOT)}}).execute()
    finally:
        if previous is None:
            os.environ.pop("FNN_WINDOWS", None)
        else:
            os.environ["FNN_WINDOWS"] = previous
    print(f"Finished {label}", flush=True)


def train(fnn_windows=DEFAULT_WINDOWS, baseline_backend=None, kernel_name="python3"):
    backend = resolve_backend(baseline_backend)
    if backend == 'gpu':
        require_gpu()
    OUT.mkdir(exist_ok=True)
    with backend_environment(backend):
        run([sys.executable, "colab_no_embeddings/evons_disinformation_detection.py"])
        execute_notebook("colab_no_embeddings/fakenewsnet_disinformation_detection.ipynb",
                         "FakeNewsNet disinformation", fnn_windows,
                         baseline_backend=backend, kernel_name=kernel_name)
        options = ["--baseline-backend", backend, "--kernel-name", kernel_name]
        run([sys.executable, "tools/run_virality_quantile_grid.py", "--dataset", "evons", *options])
        run([sys.executable, "tools/run_virality_quantile_grid.py", "--dataset", "fakenewsnet",
             *options, "--fnn-windows", fnn_windows])
    print("All training completed", flush=True)


def make_archive():
    OUT.mkdir(exist_ok=True)
    archive = OUT / "SHARE_AGGREGATES_ONLY.zip"
    # Explicit allowlist: do not accidentally package arbitrary datasets or
    # user-created CSV/JSON files that happen to be in revision_results/.
    allowed = {"preflight_manifest.json", "dataset_audit.json"}
    for label in ("fake", "real"):
        allowed.update({
            f"future_{label}.csv", f"future_{label}_summary.json",
            f"future_{label}_eligible.csv", f"future_{label}_eligible_summary.json",
            f"thresholds_{label}.csv", f"legacy_total_{label}.csv",
            f"legacy_total_{label}_summary.json",
        })
    files = [OUT / name for name in sorted(allowed) if (OUT / name).is_file()]
    # Scan generated summary files only; do not send full notebooks or tensors.
    outputs = ROOT / "colab_no_embeddings/outputs"
    files.extend(p for p in outputs.rglob("*") if p.is_file() and
                 p.suffix == ".csv" and p.name in {
                     "evons_disinformation_merged_cv_results.csv",
                     "evons_disinformation_merged_summary.csv",
                     "evons_virality_merged_cv_results.csv",
                     "evons_virality_merged_summary.csv",
                     "fakenewsnet_disinformation_merged_cv_results.csv",
                     "fakenewsnet_disinformation_merged_summary.csv",
                     "fakenewsnet_virality_merged_cv_results.csv",
                     "fakenewsnet_virality_merged_summary.csv",
                 } and (not p.name.startswith("fakenewsnet_") or
                        p.parent.name == "fakenewsnet_disinformation_prefix" or
                        p.parent.name in {"fakenewsnet_virality_prefix_q" + str(round(float(q) * 100))
                                          for q in QUANTILES}))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT))
    print(f"Ready to share: {archive} ({len(files)} aggregate files)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "audit", "train", "all", "pack"),
                        default="check")
    parser.add_argument("--fnn-windows", default=DEFAULT_WINDOWS,
                        help="FakeNewsNet windows: 1,3,5,10,full (fixed >=11-tweet cohort)")
    parser.add_argument("--baseline-backend", choices=("cpu", "gpu"),
                        default=os.environ.get("BASELINE_BACKEND", "cpu"))
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args()
    validate_windows(args.fnn_windows)
    try:
        present = check()
        audit_inputs = [REQUIRED["EVONS"][0], *REQUIRED["FakeNewsNet"][:2]]
        if args.stage in ("audit", "all") and any(not p.is_file() for p in audit_inputs):
            raise FileNotFoundError("Audits need EVONS CSV and both PolitiFact JSONLs.")
        if args.stage in ("train", "all") and not present:
            raise FileNotFoundError("Training needs all raw datasets AND the precomputed embeddings.")
        if args.stage in ("audit", "all"):
            audit()
        if args.stage in ("train", "all"):
            train(args.fnn_windows, args.baseline_backend, args.kernel_name)
        if args.stage in ("audit", "train", "all", "pack"):
            make_archive()
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "RUN_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("Failed: consult revision_results/RUN_FAILED.txt", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
