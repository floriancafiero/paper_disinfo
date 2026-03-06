#!/usr/bin/env python3
"""Synthetic one-sheet aggregate of all code paths used for report generation.

This file intentionally concatenates the source code of the scripts used in the
report pipeline into one long Python sheet for inspection and archiving.
"""

# ==============================================================================
# BEGIN: tools/evons_source_confounding_audit.py
# ==============================================================================
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple


Row = Dict[str, str]


def load_rows(path: str) -> List[Row]:
    import sys
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty CSV")
    return rows


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return float("nan")
    f1s = []
    for lab in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append((2 * p * r / (p + r)) if (p + r) else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def chi_square_source_label(rows: List[Row], source_col: str, label_col: str) -> Tuple[float, int, float]:
    sources = sorted({r[source_col] for r in rows})
    labels = sorted({r[label_col] for r in rows})

    table = {s: Counter() for s in sources}
    row_totals = Counter()
    col_totals = Counter()
    n = len(rows)

    for r in rows:
        s = r[source_col]
        y = r[label_col]
        table[s][y] += 1
        row_totals[s] += 1
        col_totals[y] += 1

    chi2 = 0.0
    for s in sources:
        for y in labels:
            obs = table[s][y]
            exp = row_totals[s] * col_totals[y] / n if n else 0.0
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

    dof = (len(sources) - 1) * (len(labels) - 1)

    # Cramer's V
    if n == 0:
        cramers_v = float("nan")
    else:
        k = min(len(sources), len(labels))
        cramers_v = math.sqrt((chi2 / n) / (k - 1)) if k > 1 else float("nan")

    return chi2, dof, cramers_v


def majority_mapping(train_rows: List[Row], source_col: str, label_col: str) -> Tuple[Dict[str, str], str]:
    by_source: Dict[str, Counter] = defaultdict(Counter)
    global_counts = Counter()
    for r in train_rows:
        by_source[r[source_col]][r[label_col]] += 1
        global_counts[r[label_col]] += 1

    source_to_label = {s: cnt.most_common(1)[0][0] for s, cnt in by_source.items()}
    global_majority = global_counts.most_common(1)[0][0]
    return source_to_label, global_majority


def predict_source_majority(test_rows: List[Row], source_to_label: Dict[str, str], fallback: str, source_col: str) -> List[str]:
    return [source_to_label.get(r[source_col], fallback) for r in test_rows]


def random_kfold_indices(n: int, k: int, seed: int) -> List[List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, v in enumerate(idx):
        folds[i % k].append(v)
    return folds


def group_kfold_indices(rows: List[Row], source_col: str, k: int, seed: int) -> List[List[int]]:
    by_source: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r[source_col]].append(i)

    groups = list(by_source.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    # greedy balance by number of rows
    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for _, row_ids in sorted(groups, key=lambda x: len(x[1]), reverse=True):
        j = min(range(k), key=lambda t: fold_sizes[t])
        folds[j].extend(row_ids)
        fold_sizes[j] += len(row_ids)
    return folds


def evaluate_cv(rows: List[Row], folds: List[List[int]], source_col: str, label_col: str, scheme: str) -> List[Dict[str, object]]:
    all_rows = list(range(len(rows)))
    out = []
    for fold_id, test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = [i for i in all_rows if i not in test_set]

        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]

        source_to_label, global_majority = majority_mapping(train_rows, source_col, label_col)
        y_true = [r[label_col] for r in test_rows]
        y_pred = predict_source_majority(test_rows, source_to_label, global_majority, source_col)

        unseen = sum(1 for r in test_rows if r[source_col] not in source_to_label)

        out.append(
            {
                "cv_scheme": scheme,
                "fold": fold_id,
                "n_test": len(test_rows),
                "unseen_source_ratio": (unseen / len(test_rows)) if test_rows else 0.0,
                "accuracy": accuracy(y_true, y_pred),
                "macro_f1": macro_f1(y_true, y_pred),
            }
        )
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def export_group_assignments(
    rows: List[Row],
    folds: List[List[int]],
    out_path: str,
    id_col: str | None,
    source_col: str,
    label_col: str,
) -> None:
    recs = []
    for fold_id, ids in enumerate(folds):
        for i in ids:
            rec = {
                "row_index": i,
                "group_fold": fold_id,
                source_col: rows[i].get(source_col, ""),
                label_col: rows[i].get(label_col, ""),
            }
            if id_col and id_col in rows[i]:
                rec[id_col] = rows[i][id_col]
            recs.append(rec)
    write_csv(out_path, recs)


def main() -> None:
    p = argparse.ArgumentParser(description="EVONS source-confounding audit")
    p.add_argument("--input", required=True, help="CSV path (e.g. evons/data/evons.csv)")
    p.add_argument("--source-col", default="media_source")
    p.add_argument("--label-col", default="is_fake")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics-out", required=True, help="Output CSV for fold metrics")
    p.add_argument("--report-out", required=True, help="Output JSON report")
    p.add_argument("--export-group-folds", default="", help="Optional CSV with row->group_fold assignment")
    p.add_argument("--id-col", default="", help="Optional row identifier column to carry in fold export")
    args = p.parse_args()

    rows = load_rows(args.input)
    required = {args.source_col, args.label_col}
    miss = required.difference(rows[0].keys())
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    labels = [r[args.label_col] for r in rows]
    sources = [r[args.source_col] for r in rows]
    source_counts = Counter(sources)
    label_counts = Counter(labels)

    chi2, dof, cramers_v = chi_square_source_label(rows, args.source_col, args.label_col)

    random_folds = random_kfold_indices(len(rows), args.folds, args.seed)
    group_folds = group_kfold_indices(rows, args.source_col, args.folds, args.seed)

    random_metrics = evaluate_cv(rows, random_folds, args.source_col, args.label_col, "random_kfold")
    group_metrics = evaluate_cv(rows, group_folds, args.source_col, args.label_col, "group_kfold_by_source")
    all_metrics = random_metrics + group_metrics

    write_csv(args.metrics_out, all_metrics)

    def avg(metric_rows: List[Dict[str, object]], key: str) -> float:
        vals = [float(r[key]) for r in metric_rows]
        return sum(vals) / len(vals) if vals else float("nan")

    report = {
        "n_rows": len(rows),
        "n_sources": len(source_counts),
        "label_distribution": dict(label_counts),
        "top_sources": source_counts.most_common(15),
        "source_label_association": {
            "chi2": chi2,
            "dof": dof,
            "cramers_v": cramers_v,
        },
        "source_only_baseline": {
            "random_kfold": {
                "mean_accuracy": avg(random_metrics, "accuracy"),
                "mean_macro_f1": avg(random_metrics, "macro_f1"),
                "mean_unseen_source_ratio": avg(random_metrics, "unseen_source_ratio"),
            },
            "group_kfold_by_source": {
                "mean_accuracy": avg(group_metrics, "accuracy"),
                "mean_macro_f1": avg(group_metrics, "macro_f1"),
                "mean_unseen_source_ratio": avg(group_metrics, "unseen_source_ratio"),
            },
        },
        "interpretation_hint": "Large drop from random_kfold to group_kfold_by_source indicates source confounding risk.",
    }

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.export_group_folds:
        export_group_assignments(
            rows,
            group_folds,
            args.export_group_folds,
            args.id_col or None,
            args.source_col,
            args.label_col,
        )

    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved report: {args.report_out}")
    if args.export_group_folds:
        print(f"Saved fold assignments: {args.export_group_folds}")
# ==============================================================================
# END: tools/evons_source_confounding_audit.py
# ==============================================================================

# ==============================================================================
# BEGIN: tools/fakenewsnet_virality_sensitivity.py
# ==============================================================================
import argparse
import csv
import json
import math
from typing import Dict, List, Sequence, Tuple


def load_propagations(jsonl_path: str) -> List[List[dict]]:
    props: List[List[dict]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and obj:
                props.append(obj)
    if not props:
        raise ValueError(f"No non-empty propagations found in {jsonl_path}")
    return props


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def auc_from_scores(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC via rank-sum (Mann-Whitney). Labels are 0/1."""
    pairs = list(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # average ranks for ties
    pairs_sorted = sorted(enumerate(pairs), key=lambda x: x[1][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs_sorted):
        j = i
        while j + 1 < len(pairs_sorted) and pairs_sorted[j + 1][1][0] == pairs_sorted[i][1][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed rank
        for k in range(i, j + 1):
            original_idx = pairs_sorted[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, lab in zip(ranks, labels) if lab == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def summarize_thresholds(totals: Sequence[float], qs: Sequence[float]) -> List[Dict[str, float]]:
    rows = []
    for q in qs:
        thr = quantile(totals, q)
        y = [1 if v >= thr else 0 for v in totals]
        pos_rate = sum(y) / len(y)
        rows.append(
            {
                "quantile": q,
                "threshold_total_likes": thr,
                "positive_rate": pos_rate,
                "n_positive": int(sum(y)),
                "n_negative": int(len(y) - sum(y)),
            }
        )
    return rows


def early_signal_rows(
    totals: Sequence[float],
    prefixes: Dict[int, List[float]],
    qs: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for q in qs:
        thr = quantile(totals, q)
        labels = [1 if t >= thr else 0 for t in totals]
        for k, pref in sorted(prefixes.items()):
            corr = pearson(pref, totals)
            auc = auc_from_scores(pref, labels)
            avg_ratio = sum((p / t) if t > 0 else 0.0 for p, t in zip(pref, totals)) / len(totals)
            rows.append(
                {
                    "quantile": q,
                    "k_prefix_tweets": k,
                    "pearson_prefix_vs_total": corr,
                    "auc_prefix_for_label": auc,
                    "avg_prefix_to_total_ratio": avg_ratio,
                }
            )
    return rows


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="FakeNewsNet virality threshold sensitivity + early-signal audit")
    ap.add_argument("--input-jsonl", required=True, help="Ordered propagation JSONL")
    ap.add_argument("--engagement-col", default="favorite_count")
    ap.add_argument("--quantiles", default="0.5,0.75,0.9,0.95")
    ap.add_argument("--k-prefix", default="1,3,5,10")
    ap.add_argument("--thresholds-out", required=True)
    ap.add_argument("--early-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    props = load_propagations(args.input_jsonl)
    qs = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
    ks = [int(x.strip()) for x in args.k_prefix.split(",") if x.strip()]

    totals: List[float] = []
    prefixes: Dict[int, List[float]] = {k: [] for k in ks}
    lengths: List[int] = []

    for seq in props:
        likes = [safe_float(t.get(args.engagement_col, 0.0)) for t in seq]
        total = sum(likes)
        totals.append(total)
        lengths.append(len(seq))
        for k in ks:
            prefixes[k].append(sum(likes[:k]))

    threshold_rows = summarize_thresholds(totals, qs)
    early_rows = early_signal_rows(totals, prefixes, qs)

    write_csv(args.thresholds_out, threshold_rows)
    write_csv(args.early_out, early_rows)

    summary = {
        "n_propagations": len(props),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "quantiles": qs,
        "k_prefix": ks,
        "files": {
            "thresholds_csv": args.thresholds_out,
            "early_csv": args.early_out,
        },
    }
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved thresholds: {args.thresholds_out}")
    print(f"Saved early-signal metrics: {args.early_out}")
    print(f"Saved summary: {args.summary_out}")
# ==============================================================================
# END: tools/fakenewsnet_virality_sensitivity.py
# ==============================================================================

# ==============================================================================
# BEGIN: tools/statistical_evaluation.py
# ==============================================================================
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
# ==============================================================================
# END: tools/statistical_evaluation.py
# ==============================================================================

# ==============================================================================
# BEGIN: tools/generate_paper_tables.py
# ==============================================================================
from pathlib import Path
import pandas as pd


def _load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    return df.sort_values(["f1", "auc"], ascending=False).reset_index(drop=True)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _esc(s: str) -> str:
    return s.replace("_", r"\_")


def build_main_table() -> str:
    evons_fake = _load_summary("colab_no_embeddings/outputs/evons_disinformation_merged_summary.csv")
    evons_vir = _load_summary("colab_no_embeddings/outputs/evons_virality_merged_summary.csv")
    fnn_fake = _load_summary("colab_no_embeddings/outputs/fakenewsnet_disinformation_merged_summary.csv")
    fnn_vir = _load_summary("colab_no_embeddings/outputs/fakenewsnet_virality_merged_summary.csv")

    blocks = [
        ("EVONS -- Fake-news Detection", evons_fake),
        ("EVONS -- Virality Prediction", evons_vir),
        ("FakeNewsNet -- Fake-news Detection", fnn_fake),
        ("FakeNewsNet -- Virality Prediction", fnn_vir),
    ]

    lines = [
        "% Auto-generated by tools/generate_paper_tables.py",
        "% Readability-oriented format, similar to the original manuscript style.",
        "% Balanced accuracy is not included because it is unavailable in consolidated summary CSVs.",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Performance comparison across datasets and tasks (10-fold CV means; readable manuscript format).}",
        "\\label{tab:readable-full-results}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Dataset / Task} & \\textbf{Model} & \\textbf{Acc} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Rec} & \\textbf{ROC-AUC} " + r"\\",
        "\\midrule",
    ]

    for section_name, df in blocks:
        lines.append(f"\\multicolumn{{7}}{{c}}{{\\textbf{{{section_name}}}}} " + r"\\")
        lines.append("\\midrule")

        best_model = df.iloc[0]["model"]
        for _, r in df.iterrows():
            model = _esc(r["model"])
            model_disp = f"\\textbf{{{model}}}" if r["model"] == best_model else model
            lines.append(
                f" & {model_disp} & {_fmt(r['accuracy'])} & {_fmt(r['f1'])} & {_fmt(r['precision'])} & {_fmt(r['recall'])} & {_fmt(r['auc'])} \\\\" 
            )
        lines.append("\\midrule")

    # replace last midrule with bottomrule
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines += ["\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def build_significance_table(stats_csv: Path) -> str:
    stats = pd.read_csv(stats_csv)
    f1 = stats[stats["metric"] == "f1"].copy()

    rows = []
    for dataset, task in [
        ("evons", "disinformation"),
        ("evons", "virality"),
        ("fakenewsnet", "disinformation"),
        ("fakenewsnet", "virality"),
    ]:
        block = f1[(f1["dataset"] == dataset) & (f1["task"] == task)].copy()
        if block.empty:
            continue

        model_scores: dict[str, float] = {}
        for _, r in block.iterrows():
            a = r["model_a"]
            b = r["model_b"]
            d = r["mean_diff"]
            model_scores[a] = model_scores.get(a, 0.0) + d
            model_scores[b] = model_scores.get(b, 0.0) - d

        best = max(model_scores, key=model_scores.get)
        comp = block[(block["model_a"] == best) | (block["model_b"] == best)].copy()
        comp["opponent"] = comp.apply(
            lambda r: r["model_b"] if r["model_a"] == best else r["model_a"], axis=1
        )
        comp["diff_best_minus_opp"] = comp.apply(
            lambda r: r["mean_diff"] if r["model_a"] == best else -r["mean_diff"], axis=1
        )
        comp = comp.sort_values("diff_best_minus_opp", ascending=False).head(3)

        for _, r in comp.iterrows():
            if r["model_a"] == best:
                ci_low = r["ci_low"]
                ci_high = r["ci_high"]
                effect_size = r["effect_size"]
            else:
                ci_low = -r["ci_high"]
                ci_high = -r["ci_low"]
                effect_size = -r["effect_size"]
            rows.append(
                {
                    "dataset_task": f"{dataset}/{task}",
                    "best": best,
                    "opponent": r["opponent"],
                    "diff": r["diff_best_minus_opp"],
                    "ci": f"[{ci_low:.4f}, {ci_high:.4f}]",
                    "p_holm": r["p_value_holm"],
                    "sig": "yes" if bool(r["significant_holm_005"]) else "no",
                    "effect_size": effect_size,
                }
            )

    out = pd.DataFrame(rows)

    lines = [
        "% Auto-generated by tools/generate_paper_tables.py",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Top pairwise F1 comparisons (best model vs strongest alternatives per dataset/task).}",
        "\\label{tab:stats-top-pairs}",
        "\\begin{tabular}{lllccccl}",
        "\\toprule",
        "Dataset/Task & Best & Compared to & $\\Delta$F1 & 95\\% CI & Holm-$p$ & Effect size & Sig. " + r"\\",
        "\\midrule",
    ]

    for _, r in out.iterrows():
        lines.append(
            f"{r['dataset_task']} & {_esc(r['best'])} & {_esc(r['opponent'])} & {r['diff']:.4f} & {r['ci']} & {r['p_holm']:.4f} & {r['effect_size']:.4f} & {r['sig']} \\\\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    Path("report/tables_enhanced_main_results.tex").write_text(build_main_table() + "\n", encoding="utf-8")
    Path("report/tables_statistical_significance.tex").write_text(
        build_significance_table(Path("report/stats_current.csv")) + "\n", encoding="utf-8"
    )
# ==============================================================================
# END: tools/generate_paper_tables.py
# ==============================================================================
