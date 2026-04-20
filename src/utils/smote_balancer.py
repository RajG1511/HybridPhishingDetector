"""
Utilities — SMOTE Class Imbalance Handler

Applies SMOTE (Synthetic Minority Over-sampling Technique) and related
strategies to address class imbalance in phishing datasets, where legitimate
emails typically outnumber phishing samples.

Public API:
    apply_smote(X, y, strategy, random_state) -> (X_res, y_res)
    report_distribution(y, label) -> None

Dependencies: imbalanced-learn (imblearn), numpy
"""

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def report_distribution(y, label: str = "") -> dict[str, int]:
    """Log and return the class distribution of a label array.

    Args:
        y: Array-like of class labels.
        label: Optional prefix for log message.

    Returns:
        Dict mapping class name to count.
    """
    dist = dict(Counter(y))
    total = sum(dist.values())
    prefix = f"[{label}] " if label else ""
    logger.info("%sClass distribution (n=%d):", prefix, total)
    for cls in sorted(dist):
        logger.info("  %-20s %6d  (%.1f%%)", cls, dist[cls], 100 * dist[cls] / total)
    return dist


def apply_smote(
    X: np.ndarray,
    y,
    strategy: str = "smote",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply over-sampling to balance class distribution.

    Only the training split should ever be passed here — never the test set.

    Available strategies:
        "smote"      — SMOTE (default)
        "adasyn"     — Adaptive synthetic sampling
        "smotetomek" — SMOTE + Tomek link under-sampling

    Args:
        X: Feature matrix of shape (n_samples, n_features). Sparse OK.
        y: Label array of shape (n_samples,).
        strategy: Resampling strategy name.
        random_state: Seed for reproducibility.
        k_neighbors: Number of nearest neighbors for SMOTE (default 5).

    Returns:
        Tuple (X_resampled, y_resampled) as numpy arrays with balanced classes.
    """
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE

    report_distribution(y, label="before SMOTE")

    if strategy == "adasyn":
        sampler = ADASYN(random_state=random_state)
    elif strategy == "smotetomek":
        sampler = SMOTETomek(random_state=random_state)
    else:
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)

    X_res, y_res = sampler.fit_resample(X, y)

    report_distribution(y_res, label=f"after {strategy.upper()}")
    return X_res, y_res
