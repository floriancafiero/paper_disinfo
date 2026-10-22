#!/usr/bin/env python3
"""Aggregate-only EVONS and FakeNewsNet/PolitiFact audit for PLOS ONE revision.

Run from the repository root. No raw tweets or article text are exported.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

QUANTILES = (0.50, 0.75, 0.90, 0.95)
PREFIXES = (1, 3, 5, 10)

# EVONS has text fields longer than the csv module default (131072 chars).
csv.field_size_limit(sys.maxsize)


def quantile(values, q):
    """Linear interpolation, matching pandas' default quantile method."""
    if not values or not 0 <= q <= 1:
        raise ValueError("Quantile requires nonempty values and q in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def thresholds(values, quantiles=QUANTILES):
    """Match main training scripts: positive iff engagement > quantile.

    Ties may make the observed positive rate differ from 1-q.
    """
    output = []
    for q in quantiles:
        t = quantile(values, q)
        positive = sum(v > t for v in values)
        output.append({
            "quantile": q, "threshold": t, "rule": ">",
            "n_evaluated": len(values), "n_positive": positive,
            "positive_rate": positive / len(values),
            "n_tied_at_threshold": sum(v == t for v in values),
        })
    return output


def finite(value, field):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return number


def audit_evons(path, quantiles=QUANTILES):
    counts = {"fake": 0, "real": 0, "missing_or_invalid": 0}
    values = []
    missing = 0
    rows = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"is_fake", "fb_engagements"}.issubset(reader.fieldnames or []):
            raise ValueError("EVONS requires is_fake and fb_engagements columns")
        for row in reader:
            rows += 1
            label = str(row["is_fake"]).strip()
            if label in ("1", "1.0"):
                counts["fake"] += 1
            elif label in ("0", "0.0"):
                counts["real"] += 1
            else:
                counts["missing_or_invalid"] += 1
            value = str(row["fb_engagements"] or "").strip()
            if not value or value.lower() in ("nan", "none", "null"):
                missing += 1
            else:
                values.append(finite(value, "fb_engagements"))
    valid = counts["fake"] + counts["real"]
    return {
        "n_rows": rows, "veracity_counts": counts,
        "fake_rate_among_valid": counts["fake"] / valid if valid else None,
        "engagement_field": "fb_engagements",
        "n_with_engagement": len(values),
        "n_missing_engagement": missing,
        "thresholds": thresholds(values, quantiles) if values else [],
        "note": "Cutoffs use observed engagements only; verify the training cohort.",
    }


def read_jsonl(path):
    sequences = []
    with Path(path).open(encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            if not line.strip():
                continue
            seq = json.loads(line)
            if not isinstance(seq, list) or any(not isinstance(p, dict) for p in seq):
                raise ValueError(f"Expected list of tweets at {path}:{i}")
            sequences.append(seq)
    return sequences


def raw_story_counts(root):
    result = {}
    for label in ("fake", "real"):
        directory = Path(root) / label
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        stories = [p for p in directory.iterdir() if p.is_dir()]
        result[label] = {
            "n_story_directories": len(stories),
            "n_directories_with_tweet_json": sum(
                (p / "tweets").is_dir() and any((p / "tweets").glob("*.json"))
                for p in stories
            ),
        }
    return result


def audit_fakenewsnet(fake_path, real_path, quantiles=QUANTILES,
                      prefixes=PREFIXES, raw_root=None):
    fake, real = read_jsonl(fake_path), read_jsonl(real_path)
    lengths, totals = [], {"fake": [], "real": []}
    missing_favorites = posts = 0
    for label, sequences in (("fake", fake), ("real", real)):
        for seq in sequences:
            lengths.append(len(seq))
            engagement = 0.
            for post in seq:
                posts += 1
                value = post.get("favorite_count")
                if value is None:
                    missing_favorites += 1
                    value = 0.  # Consistent with missing-key default in training
                engagement += finite(value, "favorite_count")
            totals[label].append(engagement)
    if not lengths:
        raise ValueError("No FakeNewsNet propagation sequences")
    pooled = totals["fake"] + totals["real"]
    result = {
        "subset": "FakeNewsNet PolitiFact (see path_creation.py)",
        "n_stories_in_processed_jsonl": len(lengths),
        "veracity_counts": {"fake": len(fake), "real": len(real)},
        "fake_rate": len(fake) / len(lengths),
        "engagement_field": "sum(tweet.favorite_count)",
        "propagation_coverage_within_processed_jsonl": {
            "n_nonempty_sequences": sum(n > 0 for n in lengths),
            "n_empty_sequences": sum(n == 0 for n in lengths),
            "n_tweets": posts,
            "min_tweets_per_story": min(lengths),
            "median_tweets_per_story": statistics.median(lengths),
            "max_tweets_per_story": max(lengths),
            "n_with_at_least_k_tweets": {str(k): sum(n >= k for n in lengths) for k in prefixes},
            "n_with_future_tweets_after_k": {str(k): sum(n > k for n in lengths) for k in prefixes},
            "n_tweets_missing_favorite_count_treated_as_zero": missing_favorites,
        },
        "thresholds": {
            "pooled_main_task": thresholds(pooled, quantiles),
            "fake_sensitivity_subset": thresholds(totals["fake"], quantiles) if fake else [],
            "real_sensitivity_subset": thresholds(totals["real"], quantiles) if real else [],
        },
        "note": "Processed JSONL coverage is NOT the original corpus coverage or API rehydration rate.",
    }
    if raw_root is not None:
        result["optional_original_story_directory_counts"] = raw_story_counts(raw_root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("both", "evons", "fakenewsnet"), default="both")
    parser.add_argument("--evons", type=Path, default=Path("evons/data/evons.csv"))
    parser.add_argument("--fake-jsonl", type=Path, default=Path("FakeNewsNet/data/ordered_fake_propagation_paths.jsonl"))
    parser.add_argument("--real-jsonl", type=Path, default=Path("FakeNewsNet/data/ordered_real_propagation_paths.jsonl"))
    parser.add_argument("--raw-politifact-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("report/dataset_audit.json"))
    args = parser.parse_args()
    result = {
        "conventions": {
            "threshold_rule": "engagement > linearly interpolated quantile",
            "quantiles": QUANTILES, "prefix_lengths": PREFIXES,
            "note": "The legacy sensitivity script uses >= rather than >; harmonize in later experiments.",
        }
    }
    if args.dataset in ("both", "evons"):
        result["EVONS"] = audit_evons(args.evons)
    if args.dataset in ("both", "fakenewsnet"):
        result["FakeNewsNet"] = audit_fakenewsnet(
            args.fake_jsonl, args.real_jsonl, raw_root=args.raw_politifact_dir
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved aggregate audit: {args.out}")


if __name__ == "__main__":
    main()
