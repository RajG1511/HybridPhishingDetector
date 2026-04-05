"""
Layer 3 — ML Base Learners

Defines and trains the base-level classifiers used in the Super Learner
ensemble: Random Forest, Linear SVM, and XGBoost. Each model exposes a
consistent fit/predict_proba interface compatible with the stacking ensemble.

Dependencies: scikit-learn, xgboost, joblib
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

_ML_DIR = MODELS_DIR / "ml"


def build_random_forest(n_estimators: int = 300, random_state: int = 42) -> RandomForestClassifier:
    """Create a Random Forest classifier with sensible defaults.

    Args:
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.

    Returns:
        Unfitted RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def build_svm(random_state: int = 42) -> CalibratedClassifierCV:
    """Create a calibrated Linear SVM (produces probability estimates).

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Unfitted CalibratedClassifierCV wrapping a LinearSVC.
    """
    svc = LinearSVC(max_iter=5000, random_state=random_state)
    return CalibratedClassifierCV(svc, cv=5)


def build_xgboost(random_state: int = 42) -> XGBClassifier:
    """Create an XGBoost classifier with sensible defaults.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Unfitted XGBClassifier instance.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def save_model(model: object, name: str) -> Path:
    """Serialize a fitted model to disk using joblib.

    Args:
        model: Fitted sklearn-compatible model.
        name: Filename stem (e.g. "random_forest").

    Returns:
        Path to the saved .joblib file.
    """
    _ML_DIR.mkdir(parents=True, exist_ok=True)
    path = _ML_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(name: str) -> object:
    """Load a serialized model from disk.

    Args:
        name: Filename stem (e.g. "random_forest").

    Returns:
        Loaded model object.
    """
    path = _ML_DIR / f"{name}.joblib"
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model
