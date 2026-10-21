"""Leakage-safe binary evaluation for severely imbalanced classifiers.

No information from the outer test set is used to select the decision cutoff.
The calibration dataset must be distinct from both model-fitting and outer-test data.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def evaluate_binary(y_true, y_score, threshold=0.5):
    """Return ROC-AUC, average precision, class prevalence and operating metrics."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_score, dtype=float).reshape(-1)
    if len(y) != len(p) or not len(y):
        raise ValueError("Scores and labels must have the same nonzero length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Requires binary labels and finite scores")
    predictions = (p >= threshold).astype(int)
    prevalence = float(y.mean())
    has_both = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "pr_baseline": prevalence,
        "majority_accuracy": max(prevalence, 1 - prevalence),
        "majority_f1": 2 * prevalence / (1 + prevalence) if prevalence > 0.5 else 0.0,
        "n_test": len(y),
        "n_positive": int(y.sum()),
        "decision_threshold": float(threshold),
    }


def tune_f1_threshold(y_cal, scores_cal):
    """Find the F1-maximizing cutoff using calibration-only labels and scores.

    Candidate thresholds include the unique scores and an always-negative sentinel.
    Tie-breaking prefers the highest threshold (more conservative predictions).
    Requires at least one positive and one negative calibration example.
    """
    y = np.asarray(y_cal, dtype=int).reshape(-1)
    p = np.asarray(scores_cal, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Invalid calibration data")
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration requires both binary classes")
    best_f1, best_threshold = -1., None
    for threshold in np.r_[np.nextafter(max(p), np.inf), np.unique(p)[::-1]]:
        f1 = f1_score(y, p >= threshold, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return best_threshold


def evaluate_with_calibration(y_test, scores_test, y_cal, scores_cal):
    """Report threshold 0.5 and independently calibrated results on same test set."""
    threshold = tune_f1_threshold(y_cal, scores_cal)
    standard = evaluate_binary(y_test, scores_test, threshold=0.5)
    calibrated = evaluate_binary(y_test, scores_test, threshold=threshold)
    return {
        **standard,
        "f1_tuned": calibrated["f1"],
        "precision_tuned": calibrated["precision"],
        "recall_tuned": calibrated["recall"],
        "accuracy_tuned": calibrated["accuracy"],
        "threshold_tuned": threshold,
        "threshold_selection": "F1 maximum on internal stratified holdout, not outer test",
    }
