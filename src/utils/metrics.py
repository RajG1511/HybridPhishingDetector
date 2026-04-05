"""
Utilities — Evaluation Metrics

Helper functions for computing standard classification metrics used throughout
the training and evaluation pipeline. All functions return plain Python dicts
for easy logging and serialization.

Dependencies: scikit-learn
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, float | str]:
    """Compute a full suite of classification metrics.

    Args:
        y_true: Ground-truth labels (integers).
        y_pred: Predicted labels (integers).
        y_proba: Optional predicted probabilities for the positive class.
            Required for AUC-ROC computation.
        labels: Optional class name list for the classification report.

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1, auc_roc (if
        y_proba provided), and report (full sklearn classification report string).
    """
    metrics: dict[str, float | str] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError as exc:
            logger.warning("AUC-ROC computation failed: %s", exc)

    logger.info(
        "Metrics — Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> None:
    """Log the confusion matrix to the logger at INFO level.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional class name list for display.
    """
    cm = confusion_matrix(y_true, y_pred)
    label_str = labels or [str(i) for i in range(cm.shape[0])]
    header = "\t".join(label_str)
    rows = "\n".join(f"{label_str[i]}\t" + "\t".join(str(v) for v in row) for i, row in enumerate(cm))
    logger.info("Confusion Matrix:\n\t%s\n%s", header, rows)
