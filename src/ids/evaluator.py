"""
Security-critical evaluation metrics for intrusion detection.

Computes comprehensive metrics for both binary and multi-class tasks,
generates visualizations, and exports structured reports.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Binary Classification Metrics                                       #
# ------------------------------------------------------------------ #

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "detection_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": int(len(y_true)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["pr_auc"] = None

    logger.info("Binary Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)

    return metrics


# ------------------------------------------------------------------ #
#  Multi-Class Metrics                                                 #
# ------------------------------------------------------------------ #

def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multi-class classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Per-class metrics
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = class_names if class_names else [str(c) for c in unique_classes]

    report = classification_report(
        y_true, y_pred,
        labels=unique_classes,
        target_names=target_names[:len(unique_classes)],
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    # Per-attack detection rate
    per_class_detection = {}
    for cls_idx, cls_name in zip(unique_classes, target_names[:len(unique_classes)]):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            detected = (y_pred[mask] == cls_idx).sum()
            per_class_detection[cls_name] = {
                "total": int(mask.sum()),
                "detected": int(detected),
                "detection_rate": float(detected / mask.sum()),
            }
    metrics["per_class_detection"] = per_class_detection

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["class_names"] = target_names[:len(unique_classes)]

    logger.info("Multi-class Metrics:")
    logger.info("  Macro F1: %.4f", metrics["macro_f1"])
    logger.info("  Weighted F1: %.4f", metrics["weighted_f1"])
    for cls_name, det in per_class_detection.items():
        logger.info("  %s: detection_rate=%.4f (%d/%d)",
                     cls_name, det["detection_rate"], det["detected"], det["total"])

    return metrics


# ------------------------------------------------------------------ #
#  Visualization                                                       #
# ------------------------------------------------------------------ #

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix."""
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    display_names = class_names[:len(unique_classes)] if class_names else [str(c) for c in unique_classes]

    fig, ax = plt.subplots(figsize=(max(8, len(unique_classes)), max(6, len(unique_classes) * 0.8)))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(display_names, fontsize=9)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str,
    title: str = "ROC Curve",
) -> None:
    """Plot and save ROC curve (binary only)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#1E88E5", lw=2, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC curve saved to %s", output_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot and save Precision-Recall curve (binary only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#43A047", lw=2, label=f"PR (AP = {ap_score:.4f})")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("PR curve saved to %s", output_path)


# ------------------------------------------------------------------ #
#  Export Functions                                                     #
# ------------------------------------------------------------------ #

def export_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Export metrics to JSON files and generate plots."""
    os.makedirs(output_dir, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    # Save metrics.json
    metrics_path = os.path.join(output_dir, f"{p}metrics.json")
    serializable = _make_serializable(metrics)
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Metrics exported to %s", metrics_path)

    # Save classification report if present
    if "classification_report" in metrics:
        report_path = os.path.join(output_dir, f"{p}classification_report.json")
        with open(report_path, "w") as f:
            json.dump(metrics["classification_report"], f, indent=2, default=str)


def run_full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task: str,
    output_dir: str,
    prefix: str = "",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline: compute metrics, generate plots, export.
    """
    os.makedirs(output_dir, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    if task == "binary":
        metrics = compute_binary_metrics(y_true, y_pred, y_proba)

        if y_proba is not None:
            try:
                plot_roc_curve(y_true, y_proba,
                               os.path.join(output_dir, f"{p}roc_curve.png"))
            except Exception as e:
                logger.warning("Could not plot ROC curve: %s", e)
            try:
                plot_pr_curve(y_true, y_proba,
                              os.path.join(output_dir, f"{p}pr_curve.png"))
            except Exception as e:
                logger.warning("Could not plot PR curve: %s", e)

        cnames = class_names or ["BENIGN", "ATTACK"]
        plot_confusion_matrix(y_true, y_pred, cnames,
                              os.path.join(output_dir, f"{p}confusion_matrix.png"))

    else:
        metrics = compute_multiclass_metrics(y_true, y_pred, class_names, y_proba)

        cnames = class_names or metrics.get("class_names", [str(i) for i in range(max(y_true) + 1)])
        plot_confusion_matrix(y_true, y_pred, cnames,
                              os.path.join(output_dir, f"{p}confusion_matrix.png"),
                              title="Multi-class Confusion Matrix")

    export_metrics(metrics, output_dir, prefix)
    return metrics


def _make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
