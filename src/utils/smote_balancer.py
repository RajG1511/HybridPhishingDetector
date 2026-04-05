"""
Utilities — SMOTE Class Imbalance Handler

Applies SMOTE (Synthetic Minority Over-sampling Technique) and related
strategies to address class imbalance in phishing datasets, where legitimate
emails typically outnumber phishing samples.

Dependencies: imbalanced-learn (imblearn)
"""

import logging

import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

logger = logging.getLogger(__name__)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply over-sampling to balance class distribution.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        strategy: One of "smote", "adasyn", or "smotetomek".
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled) with balanced classes.
    """
    original_dist = {c: int((y == c).sum()) for c in np.unique(y)}
    logger.info("Class distribution before resampling: %s", original_dist)

    if strategy == "adasyn":
        sampler = ADASYN(random_state=random_state)
    elif strategy == "smotetomek":
        sampler = SMOTETomek(random_state=random_state)
    else:
        sampler = SMOTE(random_state=random_state)

    X_res, y_res = sampler.fit_resample(X, y)
    new_dist = {c: int((y_res == c).sum()) for c in np.unique(y_res)}
    logger.info("Class distribution after %s: %s", strategy, new_dist)
    return X_res, y_res
