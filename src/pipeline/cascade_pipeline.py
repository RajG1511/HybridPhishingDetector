"""Cascade pipeline orchestration for the hybrid phishing detector."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import METADATA_URL_MODEL_PATH
from src.layer2_url.lexical_features import (
    aggregate_lexical_features,
    extract_lexical_features,
)
from src.layer2_url.url_extractor import extract_urls
from src.pipeline.email_ingester import ParsedEmail, ingest_raw
from src.pipeline.risk_scorer import RiskScorer, RiskSignals

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionResult:
    """Structured output returned by the cascade pipeline."""

    risk_score: int
    predicted_label: str
    confidence: float
    layer_outputs: dict[str, dict[str, Any]]
    explanation: str

    @property
    def verdict(self) -> str:
        """Backward-compatible alias for the final risk label."""
        return str(self.layer_outputs.get("risk", {}).get("label", "safe"))

    @property
    def layer4_used(self) -> bool:
        """Return whether Layer 4 contextual profiling was invoked."""
        return bool(self.layer_outputs.get("layer4", {}).get("used", False))


class CascadePipeline:
    """Run Layer 1, Layer 2, and Layer 3 analysis in sequence."""

    def __init__(
        self,
        layer1=None,
        layer2=None,
        layer3=None,
        layer4=None,
        shap_explainer=None,
        lime_explainer=None,
        vectorizer=None,
        *,
        narrative_generator=None,
        metadata_url_model_path: Path | str | None = METADATA_URL_MODEL_PATH,
    ) -> None:
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.shap_explainer = shap_explainer
        self.lime_explainer = lime_explainer
        self.vectorizer = vectorizer
        self.narrative_generator = narrative_generator
        self.scorer = RiskScorer(metadata_url_model_path=metadata_url_model_path)

    def run(self, email_input: ParsedEmail | bytes | str) -> DetectionResult:
        """Run the full cascade on a parsed or raw email.

        Args:
            email_input: ParsedEmail instance or raw email bytes/string.

        Returns:
            Structured detection result with layer outputs and explanation.
        """
        parsed_email = email_input if isinstance(email_input, ParsedEmail) else ingest_raw(email_input)

        layer1_output = self._run_layer1(parsed_email)
        layer2_output = self._run_layer2(parsed_email)
        layer3_output = self._run_layer3(parsed_email)

        risk_signals = RiskSignals(
            header_mismatch_count=max(
                len(layer1_output.get("header_issues", [])),
                1 if layer1_output.get("header_mismatch") else 0,
            ),
            dkim_valid=layer1_output.get("dkim") == "pass",
            spf_pass=layer1_output.get("spf") == "pass",
            spf_status=str(layer1_output.get("spf", "unknown")),
            dkim_status=str(layer1_output.get("dkim", "unknown")),
            arc_status=str(layer1_output.get("arc", "unknown")),
            protocol_risk_score=layer1_output.get("protocol_risk_score"),
            metadata_flag_count=len(layer1_output.get("metadata_flags", [])),
            url_count=int(layer2_output.get("url_count", 0)),
            url_flags=list(layer2_output.get("url_flags", [])),
            url_feature_summary=layer2_output.get("feature_summary"),
            semantic_available=bool(layer3_output.get("model_loaded", False)),
            semantic_label=str(layer3_output.get("predicted_label", "legitimate")),
            semantic_confidence=float(layer3_output.get("confidence", 0.0)),
            semantic_probabilities=layer3_output.get("class_probabilities"),
            ensemble_proba=float(layer3_output.get("phishing_probability", 0.0)),
        )
        risk_result = self.scorer.score(risk_signals)
        layer4_output = self._layer4_placeholder(risk_result.score)

        layer_outputs = {
            "layer1": layer1_output,
            "layer2": layer2_output,
            "layer3": layer3_output,
            "layer4": layer4_output,
            "risk": {
                "score": risk_result.score,
                "label": risk_result.label,
                "layer_scores": risk_result.layer_scores,
            },
        }
        explanation = self._build_explanation(layer_outputs)

        logger.info(
            "Pipeline complete: predicted_label=%s risk_score=%d verdict=%s",
            layer3_output.get("predicted_label", "legitimate"),
            risk_result.score,
            risk_result.label,
        )
        return DetectionResult(
            risk_score=risk_result.score,
            predicted_label=str(layer3_output.get("predicted_label", "legitimate")),
            confidence=float(layer3_output.get("confidence", 0.0)),
            layer_outputs=layer_outputs,
            explanation=explanation,
        )

    def _run_layer1(self, parsed_email: ParsedEmail) -> dict[str, Any]:
        """Run Layer 1 protocol checks or fall back to a neutral stub."""
        if self.layer1 is not None:
            return self._invoke_custom_layer(self.layer1, parsed_email)

        try:
            from src.layer1_protocol import analyze_protocol_authentication
        except Exception as exc:
            logger.info("Layer 1 imports unavailable, using neutral stub: %s", exc)
            return self._neutral_layer1_output()

        try:
            return analyze_protocol_authentication(parsed_email)
        except Exception as exc:
            logger.warning("Layer 1 execution failed, using neutral stub: %s", exc)
            return self._neutral_layer1_output()

    def _run_layer2(self, parsed_email: ParsedEmail) -> dict[str, Any]:
        """Run Layer 2 URL extraction and lexical analysis."""
        if self.layer2 is not None:
            return self._invoke_custom_layer(self.layer2, parsed_email)

        body = parsed_email.html_body or parsed_email.plain_body
        urls = extract_urls(body, is_html=bool(parsed_email.html_body))
        feature_summary = aggregate_lexical_features(urls)
        url_flags: list[str] = []
        per_url_features: list[dict[str, Any]] = []

        for url in urls:
            features = extract_lexical_features(url)
            per_url_features.append(features)

            if features["has_ip"]:
                url_flags.append("IP-based URL")
            if features["has_homoglyph"]:
                url_flags.append("Homoglyph URL")
            if features["suspicious_tld"]:
                url_flags.append("Suspicious TLD")
            if features["uses_shortener"]:
                url_flags.append("URL shortener")
            if features["has_at_symbol"]:
                url_flags.append("At symbol in URL")
            if features["subdomain_depth"] >= 4:
                url_flags.append("Deep subdomain nesting")
            if features["url_digit_ratio"] >= 0.15:
                url_flags.append("High digit ratio in URL")

        deduplicated_flags = list(dict.fromkeys(url_flags))
        return {
            "url_count": len(urls),
            "urls": urls,
            "feature_summary": feature_summary,
            "per_url_features": per_url_features,
            "url_flags": deduplicated_flags,
        }

    def _run_layer3(self, parsed_email: ParsedEmail) -> dict[str, Any]:
        """Run Layer 3 semantic analysis or return a neutral placeholder."""
        if self.layer3 is None:
            return self._neutral_layer3_output()

        if callable(self.layer3):
            result = self._invoke_custom_layer(self.layer3, parsed_email)
            return self._normalize_semantic_output(result)

        body = parsed_email.plain_body or parsed_email.html_body
        if not body:
            return self._neutral_layer3_output()

        try:
            from src.layer3_semantic.preprocessor import preprocess_email

            cleaned_text = preprocess_email(body)
            model_input: Any = [cleaned_text]
            if self.vectorizer is not None and hasattr(self.vectorizer, "transform"):
                model_input = self.vectorizer.transform([cleaned_text])

            if hasattr(self.layer3, "predict_proba"):
                probabilities = self.layer3.predict_proba(model_input)[0]
                classes = self._semantic_classes(probabilities)
                class_probabilities = {
                    str(classes[index]): float(probability)
                    for index, probability in enumerate(probabilities)
                }
                predicted_label = max(class_probabilities, key=class_probabilities.get)
                confidence = class_probabilities[predicted_label]
                phishing_probability = sum(
                    probability
                    for label, probability in class_probabilities.items()
                    if label != "legitimate"
                )
                return {
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "class_probabilities": class_probabilities,
                    "phishing_probability": phishing_probability,
                    "model_loaded": True,
                }

            if hasattr(self.layer3, "predict"):
                predicted_label = str(self.layer3.predict(model_input)[0])
                return {
                    "predicted_label": predicted_label,
                    "confidence": 1.0,
                    "class_probabilities": {predicted_label: 1.0},
                    "phishing_probability": 0.0 if predicted_label == "legitimate" else 1.0,
                    "model_loaded": True,
                }
        except Exception as exc:
            logger.warning("Layer 3 execution failed, using neutral output: %s", exc)

        return self._neutral_layer3_output()

    def _layer4_placeholder(self, risk_score: int) -> dict[str, Any]:
        """Return a placeholder Layer 4 status for grey-zone contextual profiling."""
        eligible = self.scorer.safe_threshold <= risk_score < self.scorer.phishing_threshold
        if eligible:
            note = (
                "Layer 4 contextual profiling would run here for a grey-zone email, "
                "but this build uses a placeholder only."
            )
            reason = "grey_zone"
        else:
            note = (
                "Layer 4 contextual profiling is only considered for grey-zone emails "
                "and was not needed for this sample."
            )
            reason = "outside_grey_zone"

        return {
            "used": False,
            "eligible": eligible,
            "status": "placeholder",
            "reason": reason,
            "grey_zone_range": {
                "low": self.scorer.safe_threshold,
                "high": self.scorer.phishing_threshold,
            },
            "note": note,
        }

    def _build_explanation(self, layer_outputs: dict[str, dict[str, Any]]) -> str:
        """Build a concise plain-English explanation string."""
        if callable(self.narrative_generator):
            try:
                return str(self.narrative_generator(layer_outputs=layer_outputs))
            except TypeError:
                logger.debug("Narrative generator did not accept layer_outputs kwarg")
            except Exception as exc:
                logger.warning("Narrative generator failed, using fallback explanation: %s", exc)

        semantic = layer_outputs["layer3"]
        risk = layer_outputs["risk"]
        protocol_flags = layer_outputs["layer1"].get("header_issues", [])
        metadata_flags = layer_outputs["layer1"].get("metadata_flags", [])
        url_flags = layer_outputs["layer2"].get("url_flags", [])

        if semantic.get("model_loaded", False):
            parts = [
                (
                    f"Semantic analysis predicts {semantic.get('predicted_label', 'legitimate')} "
                    f"with {float(semantic.get('confidence', 0.0)):.2f} confidence."
                ),
                f"Overall risk score is {risk['score']}/100, classified as {risk['label']}.",
            ]
        else:
            parts = [
                "Semantic analysis is not loaded, so this is a partial assessment based on metadata and URL signals only.",
                f"The current partial pipeline score is {risk['score']}/100.",
            ]
        if protocol_flags:
            parts.append(f"Protocol issues detected: {', '.join(protocol_flags[:3])}.")
        elif metadata_flags:
            parts.append(f"Metadata risk signals detected: {', '.join(metadata_flags[:3])}.")
        if url_flags:
            parts.append(f"URL risk signals detected: {', '.join(url_flags[:4])}.")
        return " ".join(parts)

    @staticmethod
    def _invoke_custom_layer(layer: Any, parsed_email: ParsedEmail) -> dict[str, Any]:
        """Invoke a user-supplied layer callable or analyzer object."""
        if callable(layer):
            result = layer(parsed_email)
        elif hasattr(layer, "analyze"):
            result = layer.analyze(parsed_email)
        else:
            raise TypeError("Custom layer must be callable or expose an analyze() method")

        if not isinstance(result, Mapping):
            raise TypeError("Custom layer output must be a mapping")
        return dict(result)

    @staticmethod
    def _neutral_layer1_output() -> dict[str, Any]:
        """Return a neutral Layer 1 stub."""
        return {
            "spf": "neutral",
            "dkim": "unknown",
            "arc": "unknown",
            "header_mismatch": False,
            "header_issues": [],
            "metadata_flags": [],
            "protocol_risk_score": 0,
            "metadata_features": {},
        }

    @staticmethod
    def _neutral_layer3_output() -> dict[str, Any]:
        """Return a neutral Layer 3 output when no model is available."""
        return {
            "predicted_label": "unknown",
            "confidence": 0.0,
            "class_probabilities": {},
            "phishing_probability": 0.0,
            "model_loaded": False,
        }

    @staticmethod
    def _normalize_semantic_output(result: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize a custom Layer 3 mapping into the expected schema."""
        class_probabilities = result.get("class_probabilities", {}) or {}
        if class_probabilities:
            phishing_probability = sum(
                float(probability)
                for label, probability in class_probabilities.items()
                if label != "legitimate"
            )
        else:
            confidence = float(result.get("confidence", 0.0))
            predicted_label = str(result.get("predicted_label", "legitimate"))
            phishing_probability = confidence if predicted_label != "legitimate" else 1.0 - confidence

        normalized = {
            "predicted_label": str(result.get("predicted_label", "legitimate")),
            "confidence": float(result.get("confidence", 0.0)),
            "class_probabilities": {
                str(label): float(probability)
                for label, probability in class_probabilities.items()
            },
            "phishing_probability": max(0.0, min(phishing_probability, 1.0)),
            "model_loaded": bool(result.get("model_loaded", True)),
        }
        return normalized

    def _semantic_classes(self, probabilities: Any) -> list[str]:
        """Return semantic class labels for a model probability vector."""
        classes = getattr(self.layer3, "classes_", None)
        if classes is not None:
            return [str(label) for label in classes]

        probability_count = len(probabilities)
        if probability_count == 3:
            return ["legitimate", "phishing_human", "phishing_ai"]
        if probability_count == 2:
            return ["legitimate", "phishing"]
        return [f"class_{index}" for index in range(probability_count)]


def _score_to_verdict(score: int) -> str:
    """Map a score to a final risk verdict."""
    return RiskScorer(metadata_url_model_path=METADATA_URL_MODEL_PATH).label_for_score(score)


__all__ = [
    "CascadePipeline",
    "DetectionResult",
    "_score_to_verdict",
]
