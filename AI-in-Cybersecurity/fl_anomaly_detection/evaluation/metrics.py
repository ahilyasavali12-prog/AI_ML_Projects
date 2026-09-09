"""
evaluation/metrics.py
AUC-ROC, F1, DR, FAR, convergence round computation.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
from typing import Dict, Optional


def compute_global_threshold(
    reconstruction_errors: np.ndarray,
    y_true: np.ndarray,
    percentile: float = 95.0
) -> float:
    """
    Compute anomaly threshold from reconstruction errors of NORMAL samples only.
    Threshold = percentile of reconstruction errors on normal traffic.
    """
    normal_errors = reconstruction_errors[y_true == 0]
    if len(normal_errors) == 0:
        return float(np.percentile(reconstruction_errors, percentile))
    return float(np.percentile(normal_errors, percentile))


def compute_metrics(
    reconstruction_errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    average: str = "macro"
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        reconstruction_errors : Per-sample MSE from autoencoder
        y_true                : Binary ground truth (0=normal, 1=attack)
        threshold             : Decision boundary for anomaly classification
        average               : F1 averaging strategy

    Returns:
        Dictionary with all metrics
    """
    # Binary predictions: error > threshold → anomaly
    y_pred = (reconstruction_errors > threshold).astype(int)

    # AUC-ROC (threshold-independent)
    try:
        auc = float(roc_auc_score(y_true, reconstruction_errors))
    except ValueError:
        auc = 0.5  # fallback if only one class present

    # F1 Score
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    # Confusion matrix → DR, FAR
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    dr  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0   # Detection Rate
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0   # False Alarm Rate

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "auc_roc":   auc,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "dr":        dr,
        "far":       far,
        "threshold": threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }


def find_convergence_round(
    round_metrics: list,
    metric: str = "auc_roc",
    epsilon: float = 1e-3,
    window: int = 5
) -> int:
    """
    Find the round at which training converged.
    Convergence = metric stops improving by more than epsilon over a window.

    Returns:
        Convergence round number
    """
    values = [r["metrics"][metric] for r in round_metrics]
    for i in range(window, len(values)):
        recent_improvement = max(values[i - window:i + 1]) - min(values[i - window:i + 1])
        if recent_improvement < epsilon:
            return i - window
    return len(values)  # never converged
