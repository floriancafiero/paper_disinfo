#!/usr/bin/env python3
"""Virality definition sensitivity + early-signal audit for FakeNewsNet propagations.

Uses already preprocessed propagation JSONL (ordered sequences) and does not
require embedding recomputation.

Input JSONL format: one propagation per line, each line is a JSON array of tweets.
Expected tweet field for engagement: favorite_count (default).
"""

from __future__ import annotations

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


if __name__ == "__main__":
    main()
