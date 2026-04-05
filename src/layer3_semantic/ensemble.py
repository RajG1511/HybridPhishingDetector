"""
Layer 3 — Super Learner Stacking Ensemble

Combines predictions from base learners (Random Forest, SVM, XGBoost) and
the deep learning models (BiLSTM, CharGRU) into a final phishing probability
estimate using a Logistic Regression meta-learner.

The stacking procedure uses out-of-fold predictions to avoid target leakage.

Dependencies: scikit-learn, numpy
"""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class SuperLearner:
    """Super Learner stacking meta-classifier.

    Trains a Logistic Regression meta-learner on out-of-fold base-learner
    predictions to combine multiple heterogeneous classifiers.

    Args:
        base_learners: List of unfitted sklearn-compatible classifiers that
            implement fit() and predict_proba().
        n_folds: Number of cross-validation folds for OOF generation.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        base_learners: list,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_learners = base_learners
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
        self._fitted_base: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuperLearner":
        """Fit base learners via OOF and train the meta-learner.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary label array of shape (n_samples,).

        Returns:
            Self (fitted SuperLearner).
        """
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(y), len(self.base_learners)))

        for i, learner in enumerate(self.base_learners):
            logger.info("Generating OOF predictions for base learner %d/%d", i + 1, len(self.base_learners))
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                clone = _clone_estimator(learner)
                clone.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, i] = clone.predict_proba(X[val_idx])[:, 1]

            # Refit on full training data
            learner.fit(X, y)

        self._fitted_base = self.base_learners
        self.meta_learner.fit(oof_preds, y)
        logger.info("Meta-learner fitted on OOF predictions")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the stacked ensemble.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with [P(benign), P(phishing)].
        """
        base_preds = np.column_stack(
            [learner.predict_proba(X)[:, 1] for learner in self._fitted_base]
        )
        return self.meta_learner.predict_proba(base_preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 = benign, 1 = phishing).
        """
        return self.predict_proba(X)[:, 1].round().astype(int)


def _clone_estimator(estimator: object) -> object:
    """Create a fresh clone of an sklearn estimator.

    Args:
        estimator: Fitted or unfitted sklearn-compatible estimator.

    Returns:
        New unfitted estimator with the same hyperparameters.
    """
    from sklearn.base import clone

    return clone(estimator)
