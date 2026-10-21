#!/usr/bin/env python3
"""Nadeau–Bengio corrected resampled t-tests for matched 80/20 shuffle splits.

Input: dataset,task,metric,fold,model,value. Each paired fold must use the
same data indices for both models. The test is approximate and conditional
on this pairing; no independence of overlapping folds is assumed.
"""
import argparse
import csv
import itertools
import math
from collections import defaultdict
from statistics import mean, stdev

from scipy.stats import t as student_t


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    result, previous = [0.] * len(pvalues), 0.
    for j, idx in enumerate(order):
        previous = max(previous, min(1., (len(order) - j) * pvalues[idx]))
        result[idx] = previous
    return result


def corrected_pair(a, b, train_fraction=.8):
    """Return mean, corrected SE, df and two-sided p; paired repeated holdout."""
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("At least 3 matching fold scores required")
    if not 0 < train_fraction < 1:
        raise ValueError("Train fraction must be strictly between 0 and 1")
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    # Nadeau–Bengio variance inflation relative to ordinary paired t-test.
    factor = 1 / n + (1 - train_fraction) / train_fraction
    se = stdev(d) * math.sqrt(factor)
    estimate = mean(d)
    if se == 0:
        p = 1. if estimate == 0 else 0.
        ci = (estimate, estimate)
    else:
        df = n - 1
        statistic = estimate / se
        p = float(2 * student_t.sf(abs(statistic), df))
        critical = float(student_t.ppf(.975, df))
        ci = (estimate - critical * se, estimate + critical * se)
    return {"mean_diff": estimate, "corrected_se": se, "df": n - 1,
            "p_value_corrected": p, "ci_low_corrected": ci[0],
            "ci_high_corrected": ci[1], "correction_factor": factor,
            "n_paired_splits": n}


def analyze(rows, train_fraction=.8, metric="f1"):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["metric"] != metric:
            continue
        key = (row["dataset"], row["task"], row["metric"])
        fold_id = row["fold"]
        model = row["model"]
        try:
            value = float(row["value"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        if fold_id in grouped[key][model]:
            raise ValueError(f"Duplicate (model, fold) for {key} / {model} / {fold_id}")
        grouped[key][model][fold_id] = value
    output = []
    for key, models in sorted(grouped.items()):
        family = []
        for model_a, model_b in itertools.combinations(sorted(models), 2):
            common = sorted(set(models[model_a]) & set(models[model_b]))
            if len(common) < 3:
                continue
            result = corrected_pair(
                [models[model_a][k] for k in common],
                [models[model_b][k] for k in common],
                train_fraction=train_fraction,
            )
            family.append({"dataset": key[0], "task": key[1], "metric": key[2],
                           "model_a": model_a, "model_b": model_b, **result})
        if family:
            adjusted = holm([row["p_value_corrected"] for row in family])
            for row, p in zip(family, adjusted):
                row["p_value_holm_corrected"] = p
                row["significant_holm_005"] = p < .05
                row["nominal_train_fraction"] = train_fraction
            output.extend(family)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="combined_cv.csv")
    parser.add_argument("--output", default="report/stats_corrected_resampled_f1.csv")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--train-fraction", type=float, default=.8,
                        help="Outer train fraction from the original experiment (default 0.8)")
    args = parser.parse_args()
    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"dataset", "task", "metric", "fold", "model", "value"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input needs columns {sorted(required)}")
    result = analyze(rows, args.train_fraction, args.metric)
    if not result:
        raise ValueError("No matched model comparisons")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Saved {len(result)} corrected pairwise comparisons: {args.output}")


if __name__ == "__main__":
    main()
