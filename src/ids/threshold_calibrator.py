"""
Threshold calibration for binary classification.

Optimizes the decision threshold using:
- Youden's J statistic (maximizes TPR - FPR)
- Maximum F1 score

Reports metrics before and after calibration.
"""

import os
import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def _compute_metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "detection_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def optimize_youden_j(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict]:
    """Find optimal threshold using Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])

    logger.info("Youden's J: best threshold=%.4f, J=%.4f (TPR=%.4f, FPR=%.4f)",
                best_threshold, j_scores[best_idx], tpr[best_idx], fpr[best_idx])

    metrics = _compute_metrics_at_threshold(y_true, y_proba, best_threshold)
    return best_threshold, metrics


def optimize_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict]:
    """Find optimal threshold by maximizing F1 score."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.where(
        (precision_vals[:-1] + recall_vals[:-1]) > 0,
        2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1]),
        0
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])

    logger.info("Max F1: best threshold=%.4f, F1=%.4f (P=%.4f, R=%.4f)",
                best_threshold, f1_scores[best_idx],
                precision_vals[best_idx], recall_vals[best_idx])

    metrics = _compute_metrics_at_threshold(y_true, y_proba, best_threshold)
    return best_threshold, metrics


def calibrate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run threshold calibration using configured methods.

    Returns calibration results with before/after metrics.
    """
    logger.info("=" * 60)
    logger.info("THRESHOLD CALIBRATION")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Metrics at default threshold (0.5)
    default_metrics = _compute_metrics_at_threshold(y_true, y_proba, 0.5)
    logger.info("Default threshold (0.5) metrics:")
    for k, v in default_metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)

    methods = config.get("threshold", {}).get("methods", ["youden_j", "max_f1"])
    default_method = config.get("threshold", {}).get("default_method", "youden_j")

    results = {
        "default_threshold": default_metrics,
        "calibration_methods": {},
        "recommended_method": default_method,
        "recommended_threshold": None,
    }

    for method in methods:
        if method == "youden_j":
            threshold, metrics = optimize_youden_j(y_true, y_proba)
        elif method == "max_f1":
            threshold, metrics = optimize_max_f1(y_true, y_proba)
        else:
            logger.warning("Unknown calibration method: %s", method)
            continue

        results["calibration_methods"][method] = {
            "threshold": threshold,
            "metrics": metrics,
        }

        if method == default_method:
            results["recommended_threshold"] = threshold

    # Log comparison
    logger.info("\nCalibration comparison:")
    logger.info("  Default (0.5): F1=%.4f, FPR=%.4f, FNR=%.4f",
                default_metrics["f1_score"], default_metrics["fpr"], default_metrics["fnr"])
    for method, data in results["calibration_methods"].items():
        m = data["metrics"]
        logger.info("  %s (%.4f): F1=%.4f, FPR=%.4f, FNR=%.4f",
                     method, data["threshold"], m["f1_score"], m["fpr"], m["fnr"])

    # Save
    threshold_path = os.path.join(output_dir, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Threshold calibration saved to %s", threshold_path)

    return results
