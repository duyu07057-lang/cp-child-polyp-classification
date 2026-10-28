"""Positive class is always Polyp=1; select thresholds on validation only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def select_threshold(y, scores):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if set(y) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection needs both classes and finite validation scores")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Deterministic tie break: highest threshold among equal maximum F1 values.
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def binary_metrics(y, scores, threshold):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Labels/scores must have matching nonzero lengths and finite scores")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(n=len(y), positives=int(y.sum()), negatives=int((y == 0).sum()),
                threshold=float(threshold), accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                f1=float(f1_score(y, predicted, zero_division=0)),
                auroc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, scores)) if y.sum() else 0.0,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
