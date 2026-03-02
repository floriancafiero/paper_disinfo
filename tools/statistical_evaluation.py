#!/usr/bin/env python3
"""Statistical comparison helper for CV fold metrics (stdlib-only).

Input CSV formats:
- long: dataset,task,metric,fold,model,value
- wide: dataset,task,metric,fold,<model_1>,<model_2>,... (use --wide)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from collections import defaultdict
from math import ceil, log2
from typing import Dict, Iterable, List, Sequence, Tuple


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    gt = 0
    lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n if n else float("nan")


def bootstrap_ci_mean_diff(diffs: Sequence[float], n_boot: int, alpha: float, seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int((alpha / 2) * (n_boot - 1))
    high_idx = int((1 - alpha / 2) * (n_boot - 1))
    return means[low_idx], means[high_idx]


def exact_sign_flip_pvalue(diffs: Sequence[float], mc_draws: int = 200000) -> float:
    n = len(diffs)
    observed = abs(sum(diffs) / n)

    if n <= 20:
        total = 0
        ge = 0
        for signs in itertools.product((-1, 1), repeat=n):
            total += 1
            val = abs(sum(s * d for s, d in zip(signs, diffs)) / n)
            if val >= observed:
                ge += 1
        return ge / total

    rng = random.Random(42)
    ge = 0
    for _ in range(mc_draws):
        val = abs(sum((1 if rng.random() > 0.5 else -1) * d for d in diffs) / n)
        if val >= observed:
            ge += 1
    return ge / mc_draws


def holm_bonferroni(pvals: Iterable[float]) -> List[float]:
    p = list(pvals)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * p[idx]
        if adj > running_max:
            running_max = adj
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: List[Dict[str, str]], wide: bool, id_cols: List[str], model_col: str, value_col: str, repeat_col: str) -> List[Dict[str, str]]:
    if not rows:
        raise ValueError("Input CSV is empty")

    if not wide:
        required = {"dataset", "task", "metric", "fold", model_col, value_col}
        missing = required.difference(rows[0].keys())
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        out = []
        for r in rows:
            out.append(
                {
                    "dataset": r["dataset"],
                    "task": r["task"],
                    "metric": r["metric"],
                    "fold": r["fold"],
                    "model": r[model_col],
                    "value": r[value_col],
                    "repeat": r[repeat_col] if repeat_col else "0",
                }
            )
        return out

    if len(id_cols) not in (4, 5):
        raise ValueError("--id-cols must contain 4 columns (dataset task metric fold) or 5 columns (dataset task metric fold repeat)")
    required = set(id_cols)
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing id columns for wide format: {sorted(missing)}")

    dataset_col, task_col, metric_col, fold_col = id_cols[:4]
    repeat_wide_col = id_cols[4] if len(id_cols) == 5 else ""

    model_names = [c for c in rows[0].keys() if c not in required]
    if len(model_names) < 2:
        raise ValueError("Need at least 2 model columns in wide format")

    out = []
    for r in rows:
        for m in model_names:
            out.append(
                {
                    "dataset": r[dataset_col],
                    "task": r[task_col],
                    "metric": r[metric_col],
                    "fold": r[fold_col],
                    "model": m,
                    "value": r[m],
                    "repeat": r[repeat_wide_col] if repeat_wide_col else "0",
                }
            )
    return out


def min_folds_needed_for_holm_005(family_size: int) -> int:
    if family_size <= 0:
        raise ValueError("family_size must be positive")
    return int(ceil(log2(family_size / 0.05)))


def _annotate_power_limits(rows: List[Dict[str, object]], family_size: int) -> None:
    required_folds = min_folds_needed_for_holm_005(family_size)
    for r in rows:
        n_signflip_units = int(r.get("n_signflip_units", r["n_folds"]))
        min_raw_p = 1 / (2 ** n_signflip_units)
        r["min_possible_p"] = min_raw_p
        r["holm_threshold_005"] = 0.05 / family_size
        r["holm_feasible_005"] = min_raw_p <= r["holm_threshold_005"]
        r["min_folds_for_holm_005"] = required_folds


def _apply_holm(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    adj = holm_bonferroni([float(r["p_value"]) for r in rows])
    for r, p_adj in zip(rows, adj):
        r["p_value_holm"] = p_adj
        r["significant_holm_005"] = p_adj < 0.05


def run_comparisons(rows: List[Dict[str, str]], n_boot: int, alpha: float, correction_scope: str, repeated_stratified: bool) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["task"], r["metric"])].append(r)

    results: List[Dict[str, object]] = []
    for (dataset, task, metric), group_rows in grouped.items():
        by_model_by_unit: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
        for r in group_rows:
            try:
                by_model_by_unit[r["model"]][(r["repeat"], r["fold"])] = float(r["value"])
            except ValueError:
                continue

        models = sorted(by_model_by_unit.keys())
        for a, b in itertools.combinations(models, 2):
            common_units = sorted(set(by_model_by_unit[a]).intersection(by_model_by_unit[b]))
            if len(common_units) < 3:
                continue

            arr_a = [by_model_by_unit[a][u] for u in common_units]
            arr_b = [by_model_by_unit[b][u] for u in common_units]
            diffs = [x - y for x, y in zip(arr_a, arr_b)]

            repeats = sorted({rep for rep, _ in common_units})
            diffs_for_test = diffs
            n_repeats = len(repeats)
            if repeated_stratified and n_repeats > 1:
                by_repeat: Dict[str, List[float]] = defaultdict(list)
                for (rep, _), d in zip(common_units, diffs):
                    by_repeat[rep].append(d)
                repeat_means = [sum(vals) / len(vals) for vals in by_repeat.values()]
                if len(repeat_means) >= 3:
                    diffs_for_test = repeat_means

            mean_diff = sum(diffs) / len(diffs)
            ci_low, ci_high = bootstrap_ci_mean_diff(diffs_for_test, n_boot=n_boot, alpha=alpha)
            effect = cliffs_delta(arr_a, arr_b)
            p_val = exact_sign_flip_pvalue(diffs_for_test)

            results.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "model_a": a,
                    "model_b": b,
                    "n_folds": len(common_units),
                    "n_repeats": n_repeats,
                    "n_signflip_units": len(diffs_for_test),
                    "mean_diff": mean_diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "effect_size": effect,
                    "p_value": p_val,
                }
            )

    if not results:
        raise ValueError("No valid model pairs with >=3 overlapping folds were found.")

    if correction_scope == "global":
        _apply_holm(results)
        _annotate_power_limits(results, family_size=len(results))
    elif correction_scope == "group":
        grouped_results: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for r in results:
            grouped_results[(str(r["dataset"]), str(r["task"]), str(r["metric"]))].append(r)
        for group in grouped_results.values():
            _apply_holm(group)
            _annotate_power_limits(group, family_size=len(group))
    else:
        raise ValueError("correction_scope must be 'global' or 'group'")

    results.sort(key=lambda x: (x["dataset"], x["task"], x["metric"], x["p_value_holm"]))
    return results


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "task",
        "metric",
        "model_a",
        "model_b",
        "n_folds",
        "n_repeats",
        "n_signflip_units",
        "mean_diff",
        "ci_low",
        "ci_high",
        "effect_size",
        "p_value",
        "min_possible_p",
        "p_value_holm",
        "holm_threshold_005",
        "holm_feasible_005",
        "min_folds_for_holm_005",
        "significant_holm_005",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired statistical tests for model comparisons")
    parser.add_argument("--input", required=True, help="Path to CSV input")
    parser.add_argument("--output", required=True, help="Path to CSV output")
    parser.add_argument("--wide", action="store_true", help="Input is in wide format")
    parser.add_argument("--id-cols", nargs="+", default=["dataset", "task", "metric", "fold"])
    parser.add_argument("--repeat-col", default="", help="Repeat identifier column for repeated stratified CV (long format)")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--repeated-stratified", action="store_true", help="Use repeat-level aggregation for sign-flip test when repeat info is available")
    parser.add_argument(
        "--correction-scope",
        choices=("global", "group"),
        default="group",
        help="Apply Holm correction globally across all comparisons or separately within each dataset/task/metric family",
    )
    args = parser.parse_args()

    raw = load_rows(args.input)
    tidy = normalize_rows(raw, args.wide, args.id_cols, args.model_col, args.value_col, args.repeat_col)
    out = run_comparisons(
        tidy,
        n_boot=args.n_boot,
        alpha=args.alpha,
        correction_scope=args.correction_scope,
        repeated_stratified=args.repeated_stratified,
    )
    write_csv(args.output, out)
    infeasible = sum(1 for r in out if not r["holm_feasible_005"])
    if infeasible:
        min_required = max(int(r["min_folds_for_holm_005"]) for r in out)
        print(
            f"Warning: {infeasible}/{len(out)} comparisons cannot reach Holm-corrected p<0.05 "
            "with the available number of folds (see min_possible_p and holm_threshold_005)."
        )
        print(
            f"Recommendation: use at least {min_required} folds (or repeated CV) for Holm@0.05 "
            "to be mathematically attainable in this analysis scope."
        )
    print(f"Saved {len(out)} pairwise comparisons to {args.output}")


if __name__ == "__main__":
    main()
