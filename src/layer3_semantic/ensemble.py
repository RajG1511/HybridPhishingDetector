"""
Layer 3 — Super Learner Stacking Ensemble

Combines predictions from base learners (Logistic Regression, Random Forest,
SVM, XGBoost) into a final prediction using a Logistic Regression meta-learner.

The stacking procedure uses cross_val_predict on the training set to generate
out-of-fold (OOF) probability estimates for each base learner, then trains the
meta-learner on those stacked OOF predictions to avoid target leakage.

Public API:
    SuperLearner.fit(X, y) -> self
    SuperLearner.predict_proba(X) -> ndarray
    SuperLearner.predict(X) -> ndarray
    save_ensemble(ensemble, path) -> None
    load_ensemble(path) -> SuperLearner

Dependencies: scikit-learn, numpy, joblib
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]


class SuperLearner:
    """Super Learner stacking meta-classifier (3-class).

    Generates OOF probability predictions from each base learner using
    StratifiedKFold cross-validation, then trains a Logistic Regression
    meta-learner on those stacked probabilities.

    Args:
        base_learners: Dict of {name: unfitted estimator} with predict_proba.
        n_folds: CV folds for OOF generation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        base_learners: dict,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_learners = base_learners
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = LogisticRegression(
            solver="lbfgs", max_iter=1000, n_jobs=-1
        )
        self.classes_ = CLASS_NAMES
        self._fitted_base: dict = {}

    def fit(self, X, y, val_fraction: float = 0.2) -> "SuperLearner":
        """Fit base learners and train the meta-learner using hold-out stacking.

        Splits the training data into train_stack (80%) and val_stack (20%).
        Base learners are fitted on train_stack; their predict_proba outputs on
        val_stack form the meta-features for the meta-learner. Finally all base
        learners are refit on the full dataset X, y.

        This is equivalent to 1-fold OOF stacking and is much faster than
        K-fold cross_val_predict on large datasets.

        Args:
            X: Feature matrix (n_samples, n_features). Sparse OK.
            y: Label array (n_samples,).
            val_fraction: Fraction of training data held out for meta-training.

        Returns:
            Self (fitted SuperLearner).
        """
        from sklearn.model_selection import train_test_split

        X_stack, X_val, y_stack, y_val = train_test_split(
            X, y,
            test_size=val_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        logger.info(
            "Hold-out split: stack=%d  val=%d", len(y_stack), len(y_val)
        )

        n_classes = len(np.unique(y))
        val_preds = []

        for i, (name, learner) in enumerate(self.base_learners.items()):
            logger.info(
                "Fitting base learner %s on stack set (%d/%d)...",
                name, i + 1, len(self.base_learners),
            )
            t0 = time.time()
            learner.fit(X_stack, y_stack)
            val_proba = learner.predict_proba(X_val)
            val_preds.append(val_proba)
            logger.info("  %s fitted+predicted in %.1fs", name, time.time() - t0)

        meta_X = np.hstack(val_preds)
        logger.info("Training meta-learner on hold-out stack (shape %s)...", meta_X.shape)
        self.meta_learner.fit(meta_X, y_val)
        self.classes_ = list(self.meta_learner.classes_)

        # Refit all base learners on the full training set
        logger.info("Refitting all base learners on full training set...")
        for name, learner in self.base_learners.items():
            t0 = time.time()
            learner.fit(X, y)
            self._fitted_base[name] = learner
            logger.info("  %s refit in %.1fs", name, time.time() - t0)

        logger.info("Super Learner training complete")
        return self

    def _make_stack(self, X) -> np.ndarray:
        """Build the stacked probability matrix for inference.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, n_base_learners * n_classes).
        """
        parts = [m.predict_proba(X) for m in self._fitted_base.values()]
        return np.hstack(parts)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using the stacked ensemble.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes).
        """
        stack = self._make_stack(X)
        return self.meta_learner.predict_proba(stack)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class labels (strings).
        """
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return np.array(self.classes_)[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Serialization
# ══════════════════════════════════════════════════════════════════════════════

def save_ensemble(ensemble: SuperLearner, path: str | Path) -> None:
    """Serialize the full ensemble (base models + meta-learner) to disk.

    Args:
        ensemble: Fitted SuperLearner.
        path: Directory path to save into (created if absent).
    """
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, dest / "super_learner.joblib")
    logger.info("Saved ensemble to %s", dest / "super_learner.joblib")


def load_ensemble(path: str | Path) -> SuperLearner:
    """Load a serialized SuperLearner from disk.

    Args:
        path: Directory path containing super_learner.joblib.

    Returns:
        Fitted SuperLearner.
    """
    src = Path(path) / "super_learner.joblib"
    ensemble = joblib.load(src)
    logger.info("Loaded ensemble from %s", src)
    return ensemble
