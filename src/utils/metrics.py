"""
Utilities — Evaluation Metrics

Computes and displays classification metrics for multi-class phishing detection.

Public API:
    compute_metrics(y_true, y_pred, y_proba, class_names) -> dict
    print_comparison_table(results_dict, class_names) -> None
    log_confusion_matrix(y_true, y_pred, class_names) -> None

Dependencies: scikit-learn, numpy
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

CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]


def compute_metrics(
    y_true,
    y_pred,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Compute a full suite of classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Optional (n_samples, n_classes) probability array.
            Required for AUC-ROC.
        class_names: Class names in order matching label encoding.

    Returns:
        Dict with keys: accuracy, macro_f1, macro_precision, macro_recall,
        per_class_f1, per_class_recall, per_class_precision, auc_roc,
        confusion_matrix, report.
    """
    names = class_names or CLASS_NAMES

    acc = float(accuracy_score(y_true, y_pred))
    mac_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    mac_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mac_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=names)
    per_r = recall_score(y_true, y_pred, average=None, zero_division=0, labels=names)
    per_f = f1_score(y_true, y_pred, average=None, zero_division=0, labels=names)

    cm = confusion_matrix(y_true, y_pred, labels=names)
    report = classification_report(y_true, y_pred, labels=names, target_names=names, zero_division=0)

    metrics: dict = {
        "accuracy": acc,
        "macro_precision": mac_p,
        "macro_recall": mac_r,
        "macro_f1": mac_f1,
        "per_class_precision": {n: float(per_p[i]) for i, n in enumerate(names)},
        "per_class_recall": {n: float(per_r[i]) for i, n in enumerate(names)},
        "per_class_f1": {n: float(per_f[i]) for i, n in enumerate(names)},
        "confusion_matrix": cm,
        "report": report,
        "auc_roc": None,
    }

    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            metrics["auc_roc"] = auc
        except Exception as exc:
            logger.warning("AUC-ROC computation failed: %s", exc)

    return metrics


def print_comparison_table(
    results: dict[str, dict],
    class_names: list[str] | None = None,
) -> None:
    """Print a side-by-side comparison table of model metrics.

    Args:
        results: Dict mapping model_name -> metrics dict (output of compute_metrics).
        class_names: Class names to show per-class recall for.
    """
    names = class_names or CLASS_NAMES
    models = list(results.keys())

    col_w = 14
    name_w = 22

    header = f"{'Metric':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in models)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    rows = [
        ("Accuracy", lambda m: f"{m['accuracy']:.4f}"),
        ("Macro Precision", lambda m: f"{m['macro_precision']:.4f}"),
        ("Macro Recall", lambda m: f"{m['macro_recall']:.4f}"),
        ("Macro F1", lambda m: f"{m['macro_f1']:.4f}"),
        ("AUC-ROC (OvR)", lambda m: f"{m['auc_roc']:.4f}" if m["auc_roc"] else "  N/A  "),
    ]
    for cls in names:
        short = cls.replace("phishing_", "phi_").replace("legitimate", "legit")
        rows.append((f"Recall/{short}", lambda m, c=cls: f"{m['per_class_recall'][c]:.4f}"))
        rows.append((f"F1/{short}", lambda m, c=cls: f"{m['per_class_f1'][c]:.4f}"))

    for label, fn in rows:
        line = f"{label:<{name_w}}" + "".join(f"{fn(results[m]):>{col_w}}" for m in models)
        print(line)

    print(sep)


def log_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "",
    class_names: list[str] | None = None,
) -> None:
    """Print the confusion matrix to stdout.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Optional label for the header.
        class_names: Class names in label order.
    """
    names = class_names or CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=names)

    title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
    print(f"\n{title}")
    col_w = max(len(n) for n in names) + 2
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row = f"{row_name:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in cm[i])
        print(row)
