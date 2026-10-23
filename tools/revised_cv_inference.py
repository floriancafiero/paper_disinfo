#!/usr/bin/env python3
"""Inference from the REVISED 60/20/20 resampled experiments.

Reads only aggregated per-fold results checked into this repository.
Outputs:
 - revised_cv_integrity.json: sample size, fold/paired-split integrity
 - revised_cv_model_pairs.csv: pairwise model differences WITHIN windows
 - revised_cv_prefix_vs_full.csv: paired k-vs-full differences, per model

Nadeau-Bengio variance factor sqrt(1/n + test_fraction/fit_fraction) is
used as an APPROXIMATION because threshold calibration itself uses another
20% of the cohort. Default effective fitting fraction 0.6 and test 0.2;
report this approximation and do not call it exact inference.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
METRIC = "f1_tuned"
WINDOWS = ("k1", "k3", "k5", "k10", "full")


def inputs(root=ROOT):
    base = root / "colab_no_embeddings" / "outputs"
    files = [("evons", "veracity", "NA", base / "evons_disinformation_merged_cv_results.csv"),
             ("fakenewsnet", "veracity", "NA", base / "fakenewsnet_disinformation_prefix/fakenewsnet_disinformation_merged_cv_results.csv")]
    for q in (50, 75, 90, 95):
        files.extend([
            ("evons", "virality", str(q), base / f"evons_virality_q{q}/evons_virality_merged_cv_results.csv"),
            ("fakenewsnet", "virality", str(q), base / f"fakenewsnet_virality_prefix_q{q}/fakenewsnet_virality_merged_cv_results.csv"),
        ])
    return files


def load_models(path, metric=METRIC):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"model", "fold", metric}.issubset(rows[0]):
        raise ValueError(f"Missing fold, model, or {metric}: {path}")
    bywindow = defaultdict(lambda: defaultdict(dict))
    fingerprints = defaultdict(dict)
    cohort = set()
    for row in rows:
        window = row.get("observation_window") or "article"
        model = row["model"]
        fold = row["fold"]
        val = float(row[metric])
        if not math.isfinite(val):
            raise ValueError(f"Nonfinite {metric} in {path}: {model}/{window}/{fold}")
        if fold in bywindow[window][model]:
            raise ValueError(f"Duplicate model/window/fold {model}/{window}/{fold} in {path}")
        bywindow[window][model][fold] = val
        if "cohort_fingerprint" in row:
            cohort.add(row["cohort_fingerprint"])
        if "split_fingerprint" in row:
            previous = fingerprints[window].setdefault(fold, row["split_fingerprint"])
            if row["split_fingerprint"] != previous:
                raise ValueError(f"Unpaired split at {window} fold {fold}")
    if cohort and len(cohort) != 1:
        raise ValueError(f"Inconsistent cohort fingerprints: {path}")
    expected = {str(n) for n in range(1, 11)}
    for window, models in bywindow.items():
        for model, scores in models.items():
            if set(scores) != expected:
                raise ValueError(f"{path}: {window}/{model} not 10 matching folds: {sorted(scores)}")
    if "full" in fingerprints:
        for window, fp in fingerprints.items():
            if fp != fingerprints["full"]:
                raise ValueError(f"Different splits between full and {window}: {path}")
    return bywindow, dict(cohort_fingerprint=next(iter(cohort), None),
                          windows=sorted(bywindow), models_per_window={k:len(v) for k,v in bywindow.items()},
                          n_per_model=len(expected))


def difference_stats(a, b, fit_fraction=.6, test_fraction=.2):
    if sorted(a) != sorted(b) or len(a) < 3:
        raise ValueError("Need paired metrics on >=3 identical folds")
    diffs = [a[k] - b[k] for k in sorted(a)]
    n = len(diffs)
    factor = 1 / n + test_fraction / fit_fraction
    s = stdev(diffs)
    se = s * math.sqrt(factor)
    delta = mean(diffs)
    if se == 0:
        p = 1. if delta == 0 else 0.
        lo = hi = delta
    else:
        tval = delta / se
        p = float(2 * student_t.sf(abs(tval), n - 1))
        crit = float(student_t.ppf(.975, n - 1))
        lo, hi = delta - crit * se, delta + crit * se
    return dict(mean_delta=delta, corrected_se=se, ci_low=lo, ci_high=hi,
                corrected_p=p, n_paired_splits=n, correction_factor=factor,
                paired_cohens_d=delta / s if s else float("nan"))


def adjust_holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["corrected_p"])
    previous = 0.
    for i, ix in enumerate(order):
        previous = max(previous, min(1., rows[ix]["corrected_p"] * (len(rows) - i)))
        rows[ix]["holm_p"] = previous
        rows[ix]["significant_005"] = previous < .05
        rows[ix]["family_size"] = len(rows)


def analyze(root=ROOT, metric=METRIC, fit_fraction=.6, test_fraction=.2):
    model_rows, window_rows, integrity = [], [], []
    for dataset, task, q, path in inputs(root):
        if not path.is_file():
            raise FileNotFoundError(path)
        data, check = load_models(path, metric)
        integrity.append(dict(dataset=dataset, task=task, quantile=q,
                              input=str(path.relative_to(root)), **check))
        # Test each pair of models within the SAME observation window.
        for window, models in data.items():
            family = []
            for left, right in combinations(sorted(models), 2):
                family.append(dict(dataset=dataset, task=task, quantile=q,
                                   window=window, model_a=left, model_b=right,
                                   metric=metric,
                                   **difference_stats(models[left], models[right],
                                                     fit_fraction, test_fraction)))
            adjust_holm(family)
            model_rows.extend(family)
        # Compare each early k with full, same model, same labels and same split.
        # Holm family includes all 4 prefixes times all models at this quantile.
        if "full" in data:
            family = []
            for w in WINDOWS[:-1]:
                if w not in data:
                    continue
                for model in sorted(set(data[w]) & set(data["full"])):
                    family.append(dict(dataset=dataset, task=task, quantile=q,
                                       window=w, reference_window="full", model=model,
                                       metric=metric,
                                       **difference_stats(data[w][model], data["full"][model],
                                                         fit_fraction, test_fraction)))
            adjust_holm(family)
            window_rows.extend(family)
    return model_rows, window_rows, integrity


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "report/revised_inference")
    p.add_argument("--metric", default=METRIC)
    p.add_argument("--fit-fraction", type=float, default=.6)
    p.add_argument("--test-fraction", type=float, default=.2)
    args = p.parse_args()
    if not 0 < args.fit_fraction < 1 or not 0 < args.test_fraction < 1 or args.fit_fraction + args.test_fraction > 1:
        p.error("Invalid fit/test fractions")
    pairs, windows, integrity = analyze(ROOT, args.metric, args.fit_fraction, args.test_fraction)
    target = args.output_dir
    write_csv(target / f"{args.metric}_model_pairs.csv", pairs)
    write_csv(target / f"{args.metric}_prefix_vs_full.csv", windows)
    target.mkdir(parents=True, exist_ok=True)
    (target / "integrity.json").write_text(json.dumps({
        "method": "Nadeau-Bengio approximate corrected resampled t",
        "description": "Paired scores from 10 overlapping stratified holdouts; additional calibration split not explicitly modeled",
        "fit_fraction": args.fit_fraction, "test_fraction": args.test_fraction,
        "holm_scope_model": "all pairwise models within dataset/task/quantile/window",
        "holm_scope_windows": "all 4 prefix vs full contrasts times all models within dataset/task/quantile",
        "files": integrity,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(integrity)} groups; model comparisons={len(pairs)}, prefix-vs-full={len(windows)}")
    print("Significant Holm comparisons: models=%d prefixes=%d" % (
        sum(x["significant_005"] for x in pairs), sum(x["significant_005"] for x in windows)))
    print("Prefix vs full after Holm correction:")
    for r in windows:
        if r["significant_005"]:
            print("%s %s q=%s %s vs full %s delta=%.4f Holm-p=%.5g" %
                  (r["dataset"], r["task"], r["quantile"], r["window"], r["model"],
                   r["mean_delta"], r["holm_p"]))


if __name__ == "__main__":
    main()
