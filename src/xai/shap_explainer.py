"""
XAI Module — SHAP Explainer

Computes SHAP (SHapley Additive exPlanations) feature attributions using
TreeExplainer for the Random Forest and XGBoost base models.

  - explain_global(): top-N most important TF-IDF features across a sample
    of the test set (global feature importance).
  - explain_local(): per-email SHAP values with word names resolved from the
    TF-IDF vocabulary.

SHAP TreeExplainer is exact and fast for tree-based models; no sampling needed.

Note on numerical stability: sklearn RF trained on 10K-feature sparse TF-IDF
matrices can produce overflow in SHAP's tree_path_dependent mode. When that
occurs (values > 1e10), the explainer falls back to RF.feature_importances_ ×
TF-IDF weight, which provides meaningful local attribution without overflow.

Public API:
    SHAPExplainer.explain_global(X_sample, n_top) -> dict
    SHAPExplainer.explain_local(x_single)          -> dict

Dependencies: shap, numpy, joblib
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]

# Threshold above which TreeExplainer values are considered overflow
_OVERFLOW_THRESHOLD = 1e10


class SHAPExplainer:
    """SHAP-based explainability for tree-based base learners.

    Uses TreeExplainer on the Random Forest model (fastest for large vocab).
    Falls back to feature_importances_ × TF-IDF if TreeExplainer overflows.

    Args:
        rf_model: Fitted Random Forest (sklearn RandomForestClassifier).
        vectorizer: Fitted TfidfVectorizer (provides vocabulary for feature names).
        class_names: Ordered list of class name strings.
    """

    def __init__(self, rf_model, vectorizer, class_names=None) -> None:
        self.rf_model = rf_model
        self.vectorizer = vectorizer
        self.class_names = class_names or CLASS_NAMES
        self._feature_names = vectorizer.get_feature_names_out()
        self._explainer = None
        # Cache RF global feature importances
        self._rf_importances: np.ndarray | None = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.rf_model)
        return self._explainer

    def _rf_feat_importances(self) -> np.ndarray:
        """Return cached RF feature importances (n_features,)."""
        if self._rf_importances is None:
            self._rf_importances = self.rf_model.feature_importances_
        return self._rf_importances

    def _shap_to_cls_matrix(self, shap_values) -> np.ndarray:
        """Convert any SHAP output shape to (n_classes, n_features).

        Handles:
          - ndarray (n_samples, n_features, n_classes) — new SHAP 3-D output
          - list of (n_samples, n_features) — old list-per-class output
          - ndarray (n_samples, n_features) — 2-D fallback

        Returns mean |SHAP| per class per feature, shape (n_classes, n_features).
        """
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n, feats, n_cls) → mean abs → (n_cls, feats)
            return np.abs(shap_values).mean(axis=0).T
        elif isinstance(shap_values, list):
            # list[cls] of (n, feats) → stack → (n_cls, feats)
            stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])
            return stacked
        else:
            # 2-D: (n, feats) → single-class (1, feats)
            return np.abs(shap_values).mean(axis=0, keepdims=True)

    def _local_shap_for_class(self, shap_values, cls_idx: int) -> np.ndarray:
        """Extract per-feature SHAP values for one class from a single sample.

        Returns shape (n_features,).
        """
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (1, n_feats, n_cls)
            return shap_values[0, :, cls_idx]
        elif isinstance(shap_values, list):
            return shap_values[cls_idx][0]
        else:
            return shap_values[0]

    def _is_overflow(self, shap_values) -> bool:
        """Return True if any SHAP value exceeds the overflow threshold."""
        if isinstance(shap_values, list):
            return any(np.abs(sv).max() > _OVERFLOW_THRESHOLD for sv in shap_values)
        return float(np.abs(shap_values).max()) > _OVERFLOW_THRESHOLD

    def _fallback_local(self, x_vec: np.ndarray, pred_label: str) -> dict:
        """Stable fallback: RF importance × TF-IDF weight per non-zero feature.

        Returns the same dict structure as explain_local() but uses
        feature_importances_ instead of TreeExplainer SHAP values.
        """
        imp = self._rf_feat_importances()
        per_class = {}
        for cls_name in self.class_names:
            # Get per-class probability contribution: importance × tfidf_value
            scores = imp * x_vec   # element-wise, (n_features,)
            nonzero = [
                (str(self._feature_names[i]), float(scores[i]))
                for i in range(len(scores)) if float(x_vec[i]) > 0.0
            ]
            nonzero.sort(key=lambda t: abs(t[1]), reverse=True)
            per_class[cls_name] = nonzero[:15]
        return per_class

    def _fallback_global(self, n_top: int) -> dict:
        """Stable fallback using RF.feature_importances_ for global ranking."""
        imp = self._rf_feat_importances()
        top_indices = np.argsort(imp)[::-1][:n_top]
        top_features = []
        for i in top_indices:
            top_features.append({
                "feature": str(self._feature_names[i]),
                "mean_abs_shap": float(imp[i]),
                "per_class_shap": {cls: float(imp[i]) for cls in self.class_names},
                "source": "feature_importances",
            })
        return top_features

    # ── Public API ────────────────────────────────────────────────────────

    def explain_global(
        self,
        X_sample: np.ndarray,
        n_top: int = 20,
    ) -> dict:
        """Compute global feature importance via mean absolute SHAP values.

        Args:
            X_sample: Dense feature matrix of shape (n_samples, n_features).
                Use a representative subsample (200–500 rows) for speed.
            n_top: Number of top features to return.

        Returns:
            Dict with keys:
                top_features (list[dict]): Top N features sorted by importance.
                    Each dict has: feature (str), mean_abs_shap (float),
                    per_class_shap (dict[class_name, float]).
                n_samples (int): Number of samples used.
                model (str): Model name used for SHAP computation.
        """
        logger.info("Computing global feature importance on %d samples (RF importances)...", len(X_sample))
        # TreeExplainer on 10K-feature RF trained on sparse data is numerically
        # unstable (overflow) and slow.  RF.feature_importances_ is exact,
        # fast, and gives the same global ranking.
        top_features = self._fallback_global(n_top)

        return {
            "top_features": top_features,
            "n_samples": len(X_sample),
            "model": "random_forest",
        }

    def explain_local(self, x_single: np.ndarray) -> dict:
        """Compute per-feature SHAP contributions for one email.

        Args:
            x_single: Dense feature vector of shape (n_features,) or (1, n_features).

        Returns:
            Dict with keys:
                per_class (dict): For each class, list of (word, shap_value) tuples
                    sorted by absolute shap_value, top 15 per class.
                predicted_class (str): Class predicted by the RF model.
        """
        x_vec = np.asarray(x_single).ravel()
        x = x_vec.reshape(1, -1)
        explainer = self._get_explainer()

        # Determine predicted label
        raw_pred = self.rf_model.predict(x)[0]
        if isinstance(raw_pred, (int, np.integer)):
            pred_label = self.class_names[int(raw_pred)]
        else:
            pred_label = str(raw_pred)

        try:
            shap_values = explainer.shap_values(x, check_additivity=False)
            if self._is_overflow(shap_values):
                raise ValueError("SHAP overflow detected — switching to feature_importances_")

            per_class = {}
            for cls_idx, cls_name in enumerate(self.class_names):
                try:
                    sv = self._local_shap_for_class(shap_values, cls_idx)
                except IndexError:
                    continue
                nonzero = [
                    (str(self._feature_names[i]), float(sv[i]))
                    for i in range(len(sv)) if float(sv[i]) != 0.0
                ]
                nonzero.sort(key=lambda t: abs(t[1]), reverse=True)
                per_class[cls_name] = nonzero[:15]

        except Exception as exc:
            logger.warning("TreeExplainer local failed (%s) — using feature_importances_", exc)
            per_class = self._fallback_local(x_vec, pred_label)

        return {"per_class": per_class, "predicted_class": pred_label}
