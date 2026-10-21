#!/usr/bin/env python3
"""Prospective FakeNewsNet virality audit: early likes predict ONLY later likes.

A score is the sum of favorite_count on tweets [0:k]. Its target is the
quantile-based binary label for the sum on tweets [k:end]. No model is trained.
Default cohort uses stories with > max(k) tweets for fair across-k comparison.
Run separately on the PolitiFact real and fake propagation JSONLs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Expected a nonempty list and quantile in [0, 1]")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def auc(scores, labels):
    """ROC-AUC as Mann-Whitney U / (n_positive * n_negative), with tied ranks."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_total = 0.
    j = 0
    while j < len(order):
        end = j + 1
        while end < len(order) and scores[order[end]] == scores[order[j]]:
            end += 1
        mean_rank = (j + 1 + end) / 2
        rank_total += mean_rank * sum(labels[order[t]] for t in range(j, end))
        j = end
    return (rank_total - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(scores, labels):
    """Average precision, treating identical scores as one cutoff."""
    n1 = sum(labels)
    if not n1:
        return float("nan")
    pairs = sorted(zip(scores, labels), reverse=True)
    correct = seen = 0
    ap = 0.
    j = 0
    while j < len(pairs):
        end = j + 1
        while end < len(pairs) and pairs[end][0] == pairs[j][0]:
            end += 1
        added = sum(label for _, label in pairs[j:end])
        correct += added
        seen = end
        ap += added / n1 * (correct / seen)
        j = end
    return ap


def read_sequences(path, engagement_col="favorite_count"):
    sequences = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            posts = json.loads(line)
            if not isinstance(posts, list):
                raise ValueError(f"Expected JSON array on line {line_no}")
            likes = []
            for post in posts:
                if not isinstance(post, dict):
                    raise ValueError(f"Expected tweet dictionary on line {line_no}")
                value = post.get(engagement_col, 0)  # matches existing audit
                try:
                    number = float(value) if value is not None else 0.
                except (TypeError, ValueError):
                    number = 0.
                if not math.isfinite(number):
                    number = 0.
                likes.append(number)
            sequences.append(likes)
    return sequences


def prospective_rows(sequences, ks=(1, 3, 5, 10),
                     qs=(.5, .75, .9, .95), cohort="fixed"):
    """Return (rows, coverage). Thresholds are calculated WITHIN each k's cohort.

    Exact equality is negative, matching the main training scripts (strict >).
    Fixed cohort: length > max(ks); eligible cohort: length > k at each k.
    No test/train split is needed because scores are raw engagement, not fitted.
    """
    ks = sorted(set(ks))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("k must be a positive integer")
    if cohort not in ("fixed", "eligible"):
        raise ValueError("cohort must be 'fixed' or 'eligible'")
    seqs = [s for s in sequences if isinstance(s, list)]
    eligible = {k: sum(len(s) > k for s in seqs) for k in ks}
    rows = []
    for k in ks:
        selected = [s for s in seqs if len(s) > (max(ks) if cohort == "fixed" else k)]
        if not selected:
            raise ValueError(f"No sequences with enough later tweets for k={k}")
        prefixes = [sum(s[:k]) for s in selected]
        futures = [sum(s[k:]) for s in selected]
        for q in qs:
            thr = quantile(futures, q)
            labels = [int(f > thr) for f in futures]
            n_pos = sum(labels)
            rows.append({
                "cohort": cohort, "quantile": q, "k_prefix_tweets": k,
                "n_sequences": len(selected), "n_positive": n_pos,
                "n_negative": len(selected) - n_pos,
                "positive_rate": n_pos / len(selected),
                "threshold_future_likes": thr,
                "auc_prefix_for_future_label": auc(prefixes, labels),
                "average_precision_prefix_for_future_label": average_precision(prefixes, labels),
                "ap_prevalence_baseline": n_pos / len(selected),
            })
    return rows, {
        "n_input_sequences": len(seqs),
        "n_with_future_posts_after_k": {str(k): eligible[k] for k in ks},
        "n_fixed_cohort": eligible[max(ks)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--engagement-col", default="favorite_count")
    parser.add_argument("--quantiles", default=".5,.75,.9,.95")
    parser.add_argument("--k-prefix", default="1,3,5,10")
    parser.add_argument("--cohort", choices=("fixed", "eligible"), default="fixed")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    ks = [int(s) for s in args.k_prefix.split(",")]
    qs = [float(s) for s in args.quantiles.split(",")]
    seqs = read_sequences(args.input_jsonl, args.engagement_col)
    rows, coverage = prospective_rows(seqs, ks, qs, args.cohort)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "target": "sum of favorite counts strictly AFTER first k tweets",
        "score": "sum of favorite counts over first k tweets",
        "target_threshold_rule": "future engagement > within-cohort quantile",
        "cohort": args.cohort, "quantiles": qs, "k_prefix": sorted(set(ks)),
        "coverage": coverage,
        "auc_definition": "rank-based ROC-AUC of raw prefix engagements; no fitted model",
        "average_precision_definition": "non-interpolated AP, with score ties grouped",
        "warning": "This analysis is observational: count of later tweets varies; no causal claim.",
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} prospective rows to {out}")


if __name__ == "__main__":
    main()
