"""
XAI Module — LIME Explainer

Generates local, instance-level explanations for individual email predictions
using LIME (Local Interpretable Model-agnostic Explanations). Produces
human-readable feature weights showing which words or URL features most
influenced the classification decision.

Dependencies: lime, numpy
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LIMETextExplainer:
    """LIME explainer for text-based phishing classification.

    Args:
        class_names: List of class label strings, e.g. ["benign", "phishing"].
    """

    def __init__(self, class_names: list[str] | None = None) -> None:
        self.class_names = class_names or ["benign", "phishing"]
        self._explainer = None

    def _get_explainer(self):
        """Lazily initialize the LIME text explainer."""
        if self._explainer is None:
            from lime.lime_text import LimeTextExplainer  # type: ignore[import]

            self._explainer = LimeTextExplainer(class_names=self.class_names)
        return self._explainer

    def explain_instance(
        self,
        text: str,
        predict_fn: callable,
        num_features: int = 15,
        num_samples: int = 500,
    ) -> list[tuple[str, float]]:
        """Generate a LIME explanation for a single email text.

        Args:
            text: Raw or preprocessed email body string.
            predict_fn: Callable that accepts a list of strings and returns
                an (n_samples, n_classes) probability array.
            num_features: Number of top features to include in explanation.
            num_samples: Number of perturbed samples for local fitting.

        Returns:
            List of (word, weight) tuples sorted by absolute weight descending.
            Positive weights indicate phishing signal; negative indicate benign.
        """
        explainer = self._get_explainer()
        explanation = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),  # Explain phishing class
        )
        return explanation.as_list(label=1)


class LIMETabularExplainer:
    """LIME explainer for tabular URL / protocol features.

    Args:
        feature_names: List of feature names corresponding to columns in X.
        class_names: List of class label strings.
        training_data: Representative numpy array used to estimate feature distributions.
    """

    def __init__(
        self,
        feature_names: list[str],
        class_names: list[str] | None = None,
        training_data: np.ndarray | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.class_names = class_names or ["benign", "phishing"]
        self.training_data = training_data
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import]

            self._explainer = LimeTabularExplainer(
                training_data=self.training_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                discretize_continuous=True,
            )
        return self._explainer

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        num_features: int = 10,
    ) -> list[tuple[str, float]]:
        """Generate a LIME explanation for a single feature vector.

        Args:
            instance: 1-D feature array for the email to explain.
            predict_fn: Callable accepting (n_samples, n_features) array and
                returning (n_samples, n_classes) probability array.
            num_features: Number of top features to include.

        Returns:
            List of (feature_description, weight) tuples.
        """
        explainer = self._get_explainer()
        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            labels=(1,),
        )
        return explanation.as_list(label=1)
