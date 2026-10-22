"""Text-only prefix features on a fixed FakeNewsNet cohort (>=11 tweets).

No encoder calls. Structural checks cannot establish the provenance of cached
embeddings: their identity and order must also be confirmed by the data owner.
"""
import hashlib
from datetime import datetime

DEFAULT_WINDOWS = "1,3,5,10,full"
MIN_TWEETS = 11


def validate_windows(value):
    windows = value.split(",")
    if not windows or any(w not in {"1", "3", "5", "10", "full"} for w in windows):
        raise ValueError("FNN windows must be selected from 1,3,5,10,full")
    if len(set(windows)) != len(windows):
        raise ValueError("Duplicate FNN windows are not allowed")
    return windows


def validate_sequences(paths, embeddings, category):
    """Return CPU numpy matrices, preserving the input story/tweet ordering."""
    import numpy as np

    if len(paths) != len(embeddings):
        raise ValueError(f"{category}: story count mismatch: {len(paths)} paths / {len(embeddings)} embeddings")
    matrices, dimension = [], None
    for i, (posts, vectors) in enumerate(zip(paths, embeddings)):
        where = f"{category} story {i}"
        if not isinstance(posts, list):
            raise ValueError(f"{where}: expected a tweet list")
        try:
            matrix = np.stack([np.asarray(v, dtype=np.float32) for v in vectors]) if len(vectors) else np.empty((0, 0), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: expected separate equal-sized vectors per tweet") from exc
        if matrix.ndim != 2 or (len(posts) and matrix.shape[1] == 0):
            raise ValueError(f"{where}: expected per-tweet embeddings, not an article-level pooled vector")
        if len(posts) != len(matrix):
            raise ValueError(f"{where}: tweet count mismatch: {len(posts)} tweets / {len(matrix)} embeddings")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{where}: nonfinite embeddings")
        if len(posts):
            if dimension is None:
                dimension = matrix.shape[1]
            if matrix.shape[1] != dimension:
                raise ValueError(f"{where}: inconsistent embedding dimension")
        previous = None
        for j, post in enumerate(posts):
            try:
                timestamp = datetime.strptime(post["created_at"], "%a %b %d %H:%M:%S %z %Y")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{where} tweet {j}: missing/invalid created_at") from exc
            if previous is not None and timestamp < previous:
                raise ValueError(f"{where} tweet {j}: tweets are not chronologically ordered")
            previous = timestamp
        matrices.append(matrix)
    return matrices


def prepare_features(paths, embeddings, windows, task, quantile=.5):
    """Build every window from exactly the same fake-then-real cohort.

    paths/embeddings map fake and real to story sequences. Only textual
    embeddings are features. Likes are used exclusively for total targets.
    """
    import numpy as np

    windows = validate_windows(",".join(windows))
    if task not in {"virality", "disinformation"}:
        raise ValueError("Unknown FakeNewsNet task")
    if not 0 < quantile < 1:
        raise ValueError("Quantile must lie strictly within (0,1)")
    selected, selected_posts, veracity, metadata = [], [], [], {}
    identities = []
    for category, label in (("fake", 1), ("real", 0)):
        matrices = validate_sequences(paths[category], embeddings[category], category)
        keep = [i for i, posts in enumerate(paths[category]) if len(posts) >= MIN_TWEETS]
        metadata[f"n_input_{category}"] = len(matrices)
        metadata[f"n_cohort_{category}"] = len(keep)
        metadata[f"n_excluded_{category}"] = len(matrices) - len(keep)
        for i in keep:
            selected.append(matrices[i])
            selected_posts.append(paths[category][i])
            veracity.append(label)
            identities.append((category, i, [p.get("id_str") for p in paths[category][i]]))
    if not selected:
        raise ValueError("No FakeNewsNet stories with at least 11 recovered tweets")
    if len({m.shape[1] for m in selected}) != 1:
        raise ValueError("fake/real embedding dimensions differ")
    import json
    metadata.update(
        n_cohort=len(selected), min_cohort_tweets=MIN_TWEETS,
        cohort_fingerprint=hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        target_type="total_engagement" if task == "virality" else "veracity",
    )
    y = np.asarray(veracity, dtype=int)
    if task == "virality":
        totals = []
        for i, posts in enumerate(selected_posts):
            try:
                values = np.asarray([0. if p.get("favorite_count") is None else float(p["favorite_count"]) for p in posts], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Cohort story {i}: invalid favorite_count") from exc
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Cohort story {i}: invalid favorite_count")
            totals.append(values.sum())
        if not np.isfinite(totals).all():
            raise ValueError("Nonfinite total engagement")
        threshold = float(np.quantile(totals, quantile))
        y = (np.asarray(totals) > threshold).astype(int)
        metadata.update(virality_quantile=quantile, virality_threshold=threshold)
    metadata.update(n_positive_cohort=int(y.sum()), positive_rate_cohort=float(y.mean()))
    features = {}
    for window in windows:
        stop = None if window == "full" else int(window)
        features[window] = np.stack([m[:stop].mean(axis=0, dtype=np.float64) for m in selected]).astype(np.float32)
        if not np.isfinite(features[window]).all():
            raise ValueError(f"Nonfinite pooled features for window {window}")
    return features, y, metadata


def make_splits(y, n_splits=10):
    """Reusable fit/calibration/test indices; reject single-class partitions."""
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("60/20/20 splits require both classes; do not lower the quantile automatically")
    splits = []
    try:
        outer = StratifiedShuffleSplit(n_splits=n_splits, test_size=.2, random_state=42)
        for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y), 1):
            fit, cal = train_test_split(train, test_size=.25, stratify=y[train], random_state=1000 + fold)
            if any(len(np.unique(y[idx])) != 2 for idx in (fit, cal, test)):
                raise ValueError(f"split {fold} has a single-class partition")
            splits.append((fit, cal, test))
    except ValueError as exc:
        raise ValueError(f"Insufficient class counts for 60/20/20 splits ({np.bincount(y, minlength=2).tolist()}): {exc}") from exc
    return splits


def split_fingerprint(fit, calibration, test):
    return hashlib.sha256("|".join(",".join(map(str, idx)) for idx in (fit, calibration, test)).encode()).hexdigest()
