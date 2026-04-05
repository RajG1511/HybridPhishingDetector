"""
XAI Module — SHAP Explainer

Computes SHAP (SHapley Additive exPlanations) feature attributions for
individual predictions and across the dataset. Supports both global feature
importance and local per-prediction explanations.

Dependencies: shap, numpy, matplotlib
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainability wrapper for sklearn-compatible classifiers.

    Args:
        model: Fitted classifier with a predict_proba method.
        feature_names: Optional list of feature names for display.
        explainer_type: One of "tree", "linear", or "kernel".
            "tree" is fastest for tree-based models (RF, XGBoost).
            "linear" is used for linear models (SVM, LogReg).
            "kernel" is model-agnostic but slower.
    """

    def __init__(
        self,
        model: object,
        feature_names: list[str] | None = None,
        explainer_type: str = "tree",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self._explainer = None

    def fit(self, background_data: np.ndarray) -> "SHAPExplainer":
        """Initialize the SHAP explainer with background/training data.

        Args:
            background_data: Representative sample of training features for
                baseline computation (100–200 samples recommended).

        Returns:
            Self.
        """
        import shap

        if self.explainer_type == "tree":
            self._explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "linear":
            self._explainer = shap.LinearExplainer(self.model, background_data)
        else:
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(background_data, 100)
            )
        return self

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            SHAP values array. For binary classification, shape is
            (n_samples, n_features) for the positive class.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain()")

        shap_values = self._explainer.shap_values(X)
        # For binary classifiers, return values for positive class
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def top_features(self, shap_values: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k features by mean absolute SHAP value.

        Args:
            shap_values: Output of explain(), shape (n_samples, n_features).
            top_k: Number of top features to return.

        Returns:
            List of (feature_name, mean_abs_shap) tuples, sorted descending.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[::-1][:top_k]
        names = self.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]
        return [(names[i], float(mean_abs[i])) for i in indices]
