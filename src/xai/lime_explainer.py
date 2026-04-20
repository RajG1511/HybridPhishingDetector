"""
XAI Module — LIME Text Explainer

Generates local, instance-level explanations for phishing predictions using
LIME (Local Interpretable Model-agnostic Explanations). Shows which words
most strongly pushed the prediction toward phishing or legitimate.

The classifier wrapped by LIME uses the TF-IDF → Ensemble path because LIME
perturbs the text hundreds of times; TF-IDF vectorization is fast enough
(~2ms per sample) while DistilBERT inference is too slow (~500ms).

Public API:
    LIMEExplainer.explain_prediction(raw_email_text, label_names) -> dict
    get_lime_explainer(vectorizer, ensemble) -> LIMEExplainer

Dependencies: lime, scikit-learn, numpy
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["legitimate", "phishing_human", "phishing_ai"]


class LIMEExplainer:
    """LIME text explainer wrapping the TF-IDF → Super Learner pipeline.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        ensemble: Fitted SuperLearner (or any model with predict_proba).
        class_names: Ordered list of class name strings.
        num_samples: Number of perturbed samples LIME generates per explanation.
    """

    def __init__(self, vectorizer, ensemble, class_names=None, num_samples: int = 500) -> None:
        self.vectorizer = vectorizer
        self.ensemble = ensemble
        self.class_names = class_names or CLASS_NAMES
        self.num_samples = num_samples
        self._explainer = None

    def _get_lime_explainer(self):
        if self._explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._explainer = LimeTextExplainer(
                class_names=self.class_names,
                random_state=42,
            )
        return self._explainer

    def _predict_fn(self, texts: list[str]) -> np.ndarray:
        """Predict function that LIME calls on perturbed text variants.

        Args:
            texts: List of perturbed text strings.

        Returns:
            Array of shape (n, n_classes) with probability for each class.
        """
        X = self.vectorizer.transform(texts)
        return self.ensemble.predict_proba(X)

    def explain_prediction(
        self,
        raw_text: str,
        num_features: int = 10,
        num_samples: int | None = None,
    ) -> dict:
        """Generate a LIME explanation for a single email.

        Args:
            raw_text: Preprocessed or raw email body text.
            num_features: Number of top features to include.
            num_samples: Override default LIME sample count.

        Returns:
            Dict with keys:
                predicted_label (str): Top predicted class.
                confidence (float): Probability of predicted class.
                class_probabilities (dict): {class_name: probability}.
                top_features (list[tuple[str, float]]): (word, weight) pairs
                    sorted by absolute weight. Positive weight → phishing signal,
                    negative → legitimate signal (relative to predicted class).
                predicted_class_idx (int): Integer index of predicted class.
        """
        explainer = self._get_lime_explainer()
        n_samples = num_samples or self.num_samples

        # Get prediction first
        X = self.vectorizer.transform([raw_text])
        proba = self.ensemble.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.class_names[pred_idx]
        confidence = float(proba[pred_idx])

        # Generate LIME explanation for the predicted class
        try:
            explanation = explainer.explain_instance(
                raw_text,
                self._predict_fn,
                num_features=num_features,
                num_samples=n_samples,
                labels=(pred_idx,),
            )
            features = explanation.as_list(label=pred_idx)
        except Exception as exc:
            logger.warning("LIME explanation failed: %s", exc)
            features = []

        return {
            "predicted_label": pred_label,
            "confidence": confidence,
            "class_probabilities": {
                name: float(proba[i]) for i, name in enumerate(self.class_names)
            },
            "top_features": features,
            "predicted_class_idx": pred_idx,
        }
