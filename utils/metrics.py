"""
metrics.py
----------
Standardised evaluation helpers used across training, validation, and inference.

Functions:
  - compute_metrics()       accuracy, precision, recall, F1 (macro + per-class)
  - confusion_matrix_data() raw confusion matrix as numpy array
  - print_classification_report()  human-readable per-class breakdown
  - top_k_accuracy()        top-k hit rate for multi-class problems
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger

logger = get_logger("metrics")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def confusion_matrix_data(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Build a num_classes × num_classes confusion matrix without sklearn.

    Returns:
        cm[true][pred] = count
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    return cm


def compute_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    num_classes: int,
    label_names: list[str] | None = None,
) -> dict:
    """
    Compute per-class and macro-averaged metrics from ground-truth and
    predicted integer labels.

    Args:
        y_true:       Ground-truth integer labels.
        y_pred:       Predicted integer labels (argmax of logits).
        num_classes:  Total number of classes.
        label_names:  Optional list of class names for reporting.

    Returns:
        dict with keys:
            accuracy, macro_precision, macro_recall, macro_f1,
            per_class: [{name, precision, recall, f1, support}, ...]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix_data(y_true, y_pred, num_classes)

    per_class = []
    precisions, recalls, f1s = [], [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(cm[i, :].sum())
        name = label_names[i] if label_names and i < len(label_names) else str(i)

        per_class.append(
            {
                "name": name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )

        if support > 0:   # exclude zero-support classes from macro average
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    accuracy = float((y_true == y_pred).sum() / len(y_true))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "macro_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "per_class": per_class,
    }


def top_k_accuracy(
    y_true: list | np.ndarray,
    y_prob: np.ndarray,
    k: int = 5,
) -> float:
    """
    Fraction of samples where the true label appears in the top-k predictions.

    Args:
        y_true: Ground-truth integer labels, shape (N,).
        y_prob: Class probability array, shape (N, num_classes).
        k:      Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]  # (N, k)
    hits = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return round(hits / len(y_true), 4)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_classification_report(metrics: dict) -> None:
    """Pretty-print the output of compute_metrics()."""
    header = f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    separator = "-" * len(header)

    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    for pc in metrics["per_class"]:
        logger.info(
            f"{pc['name']:<22} {pc['precision']:>10.4f} {pc['recall']:>10.4f}"
            f" {pc['f1']:>10.4f} {pc['support']:>10}"
        )

    logger.info(separator)
    logger.info(
        f"{'MACRO AVG':<22} {metrics['macro_precision']:>10.4f}"
        f" {metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f}"
    )
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(separator)
