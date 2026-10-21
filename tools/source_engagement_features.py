"""Train-only, leave-one-out source engagement features for EVONS.

For model-fit item i from source s, its feature excludes engagement_i.
For validation and test examples, features use only model-fit engagements.
Unseen or singleton fit sources use the mean of other fit examples.
"""
import numpy as np


def source_engagement_features(source_fit, engagement_fit, source_other):
    sf = np.asarray(source_fit)
    ef = np.asarray(engagement_fit, dtype=float)
    so = np.asarray(source_other)
    if sf.ndim != 1 or ef.ndim != 1 or len(sf) != len(ef) or not len(sf):
        raise ValueError("Expected nonempty aligned fit source and engagement vectors")
    if not np.isfinite(ef).all():
        raise ValueError("Nonfinite training engagement: clean before cross-validation")
    sums = {}
    counts = {}
    for source, value in zip(sf, ef):
        sums[source] = sums.get(source, 0.) + value
        counts[source] = counts.get(source, 0) + 1
    all_total = float(ef.sum())
    n = len(ef)
    global_mean = all_total / n
    fit_values = []
    for source, value in zip(sf, ef):
        if counts[source] > 1:
            fit_values.append((sums[source] - value) / (counts[source] - 1))
        elif n > 1:
            fit_values.append((all_total - value) / (n - 1))
        else:
            fit_values.append(0.)
    other_values = [sums[s] / counts[s] if s in sums else global_mean for s in so]
    fit_values = np.asarray(fit_values, dtype=np.float32)
    other_values = np.asarray(other_values, dtype=np.float32)
    mean, std = fit_values.mean(), fit_values.std()
    denom = max(float(std), 1e-6)
    return (fit_values - mean) / denom, (other_values - mean) / denom
