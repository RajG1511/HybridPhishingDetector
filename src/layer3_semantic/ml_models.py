"""
Layer 3 — ML Base Learners

Defines, trains, and serializes the four base-level classifiers used in the
Super Learner ensemble:
    - Logistic Regression (baseline, solver=lbfgs)
    - Random Forest       (n_estimators=200)
    - LinearSVC           (wrapped in CalibratedClassifierCV for probabilities)
    - XGBoost             (n_estimators=200, max_depth=6)

All models expose a consistent fit / predict / predict_proba interface.
Models are serialized to models/ml/ using joblib.

Dependencies: scikit-learn, xgboost, joblib
"""

import logging
from pathlib import Path

import joblib
import numpy as np

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

_ML_DIR = MODELS_DIR / "ml"

# Canonical class order for all models
CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]


# ══════════════════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════════════════

def build_logistic_regression(random_state: int = 42):
    """Build a Logistic Regression classifier.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Unfitted LogisticRegression.
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=random_state,
        n_jobs=-1,
    )


def build_random_forest(n_estimators: int = 200, random_state: int = 42):
    """Build a Random Forest classifier.

    Args:
        n_estimators: Number of trees.
        random_state: Seed for reproducibility.

    Returns:
        Unfitted RandomForestClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def build_svm(random_state: int = 42):
    """Build a calibrated Linear SVM (produces probability estimates).

    LinearSVC is much faster than SVC on large sparse datasets. Wrapped in
    CalibratedClassifierCV to expose predict_proba.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Unfitted CalibratedClassifierCV wrapping LinearSVC.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    svc = LinearSVC(max_iter=5000, random_state=random_state, C=1.0)
    return CalibratedClassifierCV(svc, cv=3)


def build_xgboost(n_estimators: int = 200, random_state: int = 42):
    """Build an XGBoost classifier wrapped to handle string labels.

    XGBoost requires integer labels; the wrapper encodes string labels to
    integers at fit-time and decodes predictions back to strings.

    Args:
        n_estimators: Number of boosting rounds.
        random_state: Seed for reproducibility.

    Returns:
        Unfitted LabelEncodingXGBWrapper.
    """
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",  # MUCH faster for 10k+ features
        n_jobs=2,            # Matches Colab free tier exactly
        verbosity=0,
    )
    return LabelEncodingXGBWrapper(xgb)


class LabelEncodingXGBWrapper:
    """Thin wrapper around XGBClassifier that handles string class labels.

    XGBoost only accepts integer labels. This wrapper encodes string labels
    to integers at fit-time and decodes predictions back to strings, giving
    the same predict/predict_proba interface as sklearn classifiers.

    Inherits from BaseEstimator and ClassifierMixin so it is fully compatible
    with sklearn utilities (cross_val_predict, Pipeline, GridSearchCV, etc.).

    Args:
        xgb_model: Unfitted XGBClassifier instance.
    """

    from sklearn.base import BaseEstimator, ClassifierMixin

    def __init__(self, xgb_model) -> None:
        self.xgb_model = xgb_model
        self._le = None
        self.classes_ = None

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.multiclass import unique_labels

        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = list(self._le.classes_)
        self.xgb_model.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.xgb_model.predict(X)
        return self._le.inverse_transform(y_enc)

    def predict_proba(self, X):
        # XGBClassifier returns probabilities ordered by encoded integer labels
        # (which matches alphabetical order of string labels after LabelEncoder)
        return self.xgb_model.predict_proba(X)

    def get_params(self, deep=True):
        return {"xgb_model": self.xgb_model}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags
        tags = Tags()
        tags.estimator_type = "classifier"
        return tags


def get_all_base_learners(random_state: int = 42) -> dict:
    """Return a dict of name -> unfitted model for all base learners.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Ordered dict: {model_name: unfitted_model}
    """
    return {
        "logistic_regression": build_logistic_regression(random_state),
        "random_forest": build_random_forest(random_state=random_state),
        "svm": build_svm(random_state),
        "xgboost": build_xgboost(random_state=random_state),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, model_name: str = ""):
    """Fit a model and log timing.

    Args:
        model: Unfitted sklearn-compatible model.
        X_train: Training feature matrix (dense or sparse).
        y_train: Training labels.
        model_name: Label for logging.

    Returns:
        Fitted model.
    """
    import time

    label = model_name or type(model).__name__
    logger.info("Training %s on %d samples...", label, len(y_train))
    t0 = time.time()
    model.fit(X_train, y_train)
    logger.info("%s trained in %.1fs", label, time.time() - t0)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Serialization
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, name: str, subdir: str = "") -> Path:
    """Serialize a fitted model to disk using joblib.

    Args:
        model: Fitted sklearn-compatible model.
        name: Filename stem (e.g. "random_forest").
        subdir: Optional subdirectory under models/ml/ (e.g. "ensemble").

    Returns:
        Path to the saved .joblib file.
    """
    dest = _ML_DIR / subdir if subdir else _ML_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Saved %s to %s", name, path)
    return path


def load_model(name: str, subdir: str = "") -> object:
    """Load a serialized model from disk.

    Args:
        name: Filename stem (e.g. "random_forest").
        subdir: Optional subdirectory under models/ml/.

    Returns:
        Loaded model object.
    """
    src = (_ML_DIR / subdir / f"{name}.joblib") if subdir else (_ML_DIR / f"{name}.joblib")
    model = joblib.load(src)
    logger.info("Loaded %s from %s", name, src)
    return model
