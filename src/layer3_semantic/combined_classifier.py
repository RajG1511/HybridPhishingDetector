"""
Layer 3 — Combined Meta-Meta-Classifier

Fuses the TF-IDF Super Learner ensemble and the BiLSTM (DistilBERT) model
by stacking their 3-class probability outputs into a 6-dimensional feature
vector per email, then training a Logistic Regression meta-meta-learner on
those fused probabilities.

Neither base model is retrained. The meta-meta-learner is fitted on held-out
predictions: we use a fresh portion of the test set that neither base model
saw during training (valid because both base models were trained only on the
training split).

Public API:
    build_combined_classifier(...)  -> (CombinedClassifier, metrics_dict)
    CombinedClassifier.predict_proba(tfidf_X, bert_X) -> ndarray
    CombinedClassifier.predict(tfidf_X, bert_X)       -> ndarray
    save_combined / load_combined
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)

_ENSEMBLE_DIR = MODELS_DIR / "ml" / "ensemble"
CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]


class CombinedClassifier:
    """Meta-meta-classifier fusing TF-IDF ensemble and BiLSTM probabilities.

    At inference time, both modalities must be provided. The meta-learner
    receives a (n_samples, 6) stack: [ensemble_3_proba | bilstm_3_proba].

    Args:
        ensemble: Fitted SuperLearner (TF-IDF path).
        bilstm: Fitted BiLSTMClassifier (DistilBERT path).
        idx_to_label: Mapping from integer index to class string.
        device: Torch device for BiLSTM inference.
    """

    def __init__(self, ensemble, bilstm, idx_to_label: dict, device=None) -> None:
        self.ensemble = ensemble
        self.bilstm = bilstm
        self.idx_to_label = idx_to_label
        self.device = device
        self.meta = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        self.classes_ = CLASS_NAMES

    def _stack(self, tfidf_X, bert_X: np.ndarray) -> np.ndarray:
        """Generate stacked 6-dim probability matrix.

        Args:
            tfidf_X: TF-IDF sparse/dense feature matrix (n, 10000).
            bert_X: DistilBERT embedding matrix (n, 768).

        Returns:
            Numpy array of shape (n, 6).
        """
        import torch, torch.nn.functional as F

        p_ensemble = self.ensemble.predict_proba(tfidf_X)  # (n, 3)

        # BiLSTM inference
        device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bilstm.eval().to(device)
        X_t = torch.tensor(bert_X, dtype=torch.float32).unsqueeze(1)
        parts = []
        with torch.no_grad():
            for i in range(0, len(X_t), 256):
                logits = self.bilstm(X_t[i : i + 256].to(device))
                parts.append(F.softmax(logits, dim=1).cpu().numpy())
        p_bilstm = np.vstack(parts)  # (n, 3)

        return np.hstack([p_ensemble, p_bilstm])  # (n, 6)

    def fit(self, tfidf_X, bert_X: np.ndarray, y) -> "CombinedClassifier":
        """Fit the meta-meta-learner on held-out stacked probabilities.

        Args:
            tfidf_X: TF-IDF features for the meta-training split.
            bert_X: BERT embeddings for the meta-training split.
            y: Labels for the meta-training split.

        Returns:
            Self.
        """
        stack = self._stack(tfidf_X, bert_X)
        self.meta.fit(stack, y)
        self.classes_ = list(self.meta.classes_)
        logger.info("Meta-meta-learner fitted on %d samples (6-dim stack)", len(y))
        return self

    def predict_proba(self, tfidf_X, bert_X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Returns:
            Array of shape (n, 3).
        """
        return self.meta.predict_proba(self._stack(tfidf_X, bert_X))

    def predict(self, tfidf_X, bert_X: np.ndarray) -> np.ndarray:
        """Predict class labels (strings).

        Returns:
            Array of predicted class name strings.
        """
        proba = self.predict_proba(tfidf_X, bert_X)
        idx = np.argmax(proba, axis=1)
        return np.array(self.classes_)[idx]


def save_combined(clf: CombinedClassifier, path: str | Path | None = None) -> Path:
    """Serialize the combined classifier to disk.

    Args:
        clf: Fitted CombinedClassifier.
        path: Destination .joblib path.

    Returns:
        Path to saved file.
    """
    dest = Path(path) if path else (_ENSEMBLE_DIR / "final_combined_classifier.joblib")
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, dest)
    logger.info("Saved combined classifier to %s", dest)
    return dest


def load_combined(path: str | Path | None = None) -> CombinedClassifier:
    """Load a serialized CombinedClassifier from disk."""
    src = Path(path) if path else (_ENSEMBLE_DIR / "final_combined_classifier.joblib")
    clf = joblib.load(src)
    logger.info("Loaded combined classifier from %s", src)
    return clf
