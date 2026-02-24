#!/usr/bin/env python3
"""Audit source-confounding risk on EVONS-style CSV and generate group folds.

No embedding recomputation required.

Expected columns:
- source column (default: media_source)
- label column  (default: label)
Optional id column for fold export.

Outputs:
1) JSON report with source-label association diagnostics
2) CSV with per-fold scores for:
   - random_kfold source-only baseline
   - group_kfold(by source) source-only baseline
3) (optional) CSV assigning each row to a group fold for reuse in notebooks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple


Row = Dict[str, str]


def load_rows(path: str) -> List[Row]:
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


def export_group_assignments(rows: List[Row], folds: List[List[int]], out_path: str, id_col: str | None) -> None:
    recs = []
    for fold_id, ids in enumerate(folds):
        for i in ids:
            rec = {
                "row_index": i,
                "group_fold": fold_id,
                "media_source": rows[i].get("media_source", ""),
                "label": rows[i].get("label", ""),
            }
            if id_col and id_col in rows[i]:
                rec[id_col] = rows[i][id_col]
            recs.append(rec)
    write_csv(out_path, recs)


def main() -> None:
    p = argparse.ArgumentParser(description="EVONS source-confounding audit")
    p.add_argument("--input", required=True, help="CSV path (e.g. evons/data/evons.csv)")
    p.add_argument("--source-col", default="media_source")
    p.add_argument("--label-col", default="label")
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
        export_group_assignments(rows, group_folds, args.export_group_folds, args.id_col or None)

    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved report: {args.report_out}")
    if args.export_group_folds:
        print(f"Saved fold assignments: {args.export_group_folds}")


if __name__ == "__main__":
    main()
