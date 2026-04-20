"""Learned metadata + URL phishing model utilities.

This module provides a lightweight wrapper around a saved binary classifier
that consumes the engineered Layer 1/2 signals and returns a phishing
probability. The rule-based scorer can use this model when present and fall
back gracefully when it is absent.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import joblib

from config.settings import METADATA_URL_MODEL_PATH

if TYPE_CHECKING:
    from src.pipeline.risk_scorer import RiskSignals

logger = logging.getLogger(__name__)

# Full feature set — used by feature extraction scripts and CSV output.
# Includes all protocol one-hot encodings for completeness.
DEFAULT_FEATURE_NAMES = [
    "protocol_risk_score",
    "header_mismatch_count",
    "metadata_flag_count",
    "url_count",
    "url_flag_count",
    "url_length_mean",
    "url_length_max",
    "url_entropy_mean",
    "url_entropy_max",
    "has_ip_max",
    "has_homoglyph_max",
    "suspicious_tld_max",
    "uses_shortener_max",
    "has_at_symbol_max",
    "subdomain_depth_mean",
    "subdomain_depth_max",
    "path_depth_mean",
    "path_depth_max",
    "num_special_chars_mean",
    "num_special_chars_max",
    "num_url_params_mean",
    "num_url_params_max",
    "url_digit_ratio_mean",
    "url_digit_ratio_max",
    "spf_pass",
    "spf_fail",
    "spf_softfail",
    "spf_none",
    "spf_unknown",
    "dkim_pass",
    "dkim_fail",
    "dkim_missing",
    "dkim_unknown",
    "arc_pass",
    "arc_fail",
    "arc_missing",
    "arc_unknown",
]



@dataclass(slots=True)
class MetadataURLModel:
    """Thin wrapper around a saved metadata + URL phishing classifier."""

    model: Any
    feature_names: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_NAMES))
    threshold: float = 0.5
    model_name: str = "metadata_url_model"
    training_metrics: dict[str, Any] = field(default_factory=dict)

    def predict_probability(self, signals: "RiskSignals") -> float:
        """Return the phishing probability for a RiskSignals instance."""
        feature_dict = build_feature_dict_from_signals(signals)
        vector = [[float(feature_dict.get(name, 0.0)) for name in self.feature_names]]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            if len(probabilities) == 1:
                return _clip_probability(probabilities[0])
            return _clip_probability(probabilities[-1])

        if hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(vector)[0])
            return _clip_probability(1.0 / (1.0 + math.exp(-decision)))

        prediction = float(self.model.predict(vector)[0])
        return _clip_probability(prediction)

    def predict_label(self, signals: "RiskSignals") -> str:
        """Return a binary label from the saved model threshold."""
        return "phishing" if self.predict_probability(signals) >= self.threshold else "legitimate"

    def save(self, path: Path | str = METADATA_URL_MODEL_PATH) -> Path:
        """Persist the model bundle to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "training_metrics": self.training_metrics,
        }
        joblib.dump(payload, output_path)
        return output_path

    @classmethod
    def load(cls, path: Path | str = METADATA_URL_MODEL_PATH) -> "MetadataURLModel":
        """Load a saved metadata + URL model bundle from disk."""
        payload = joblib.load(Path(path))
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("Saved metadata + URL model must be a mapping or MetadataURLModel")

        return cls(
            model=payload["model"],
            feature_names=list(payload.get("feature_names", DEFAULT_FEATURE_NAMES)),
            threshold=float(payload.get("threshold", 0.5)),
            model_name=str(payload.get("model_name", "metadata_url_model")),
            training_metrics=dict(payload.get("training_metrics", {})),
        )


def maybe_load_metadata_url_model(
    path: Path | str = METADATA_URL_MODEL_PATH,
) -> MetadataURLModel | None:
    """Load a metadata + URL model if the artifact exists."""
    model_path = Path(path)
    if not model_path.exists():
        return None

    try:
        return MetadataURLModel.load(model_path)
    except Exception as exc:
        logger.warning("Could not load metadata + URL model at %s: %s", model_path, exc)
        return None


def build_feature_dict_from_signals(signals: "RiskSignals") -> dict[str, float]:
    """Convert RiskSignals into the numeric feature vector expected by the model."""
    summary = signals.url_feature_summary or {}

    feature_dict = {name: 0.0 for name in DEFAULT_FEATURE_NAMES}
    feature_dict.update(
        {
            "protocol_risk_score": _bounded_float(signals.protocol_risk_score, upper=100.0),
            "header_mismatch_count": max(0.0, float(signals.header_mismatch_count)),
            "metadata_flag_count": max(0.0, float(signals.metadata_flag_count)),
            "url_count": max(0.0, float(signals.url_count)),
            "url_flag_count": max(0.0, float(len(signals.url_flags))),
            "url_length_mean": _bounded_float(summary.get("url_length_mean"), upper=10000.0),
            "url_length_max": _bounded_float(summary.get("url_length_max"), upper=10000.0),
            "url_entropy_mean": _bounded_float(summary.get("url_entropy_mean"), upper=10.0),
            "url_entropy_max": _bounded_float(summary.get("url_entropy_max"), upper=10.0),
            "has_ip_max": _bounded_float(summary.get("has_ip_max"), upper=1.0),
            "has_homoglyph_max": _bounded_float(summary.get("has_homoglyph_max"), upper=1.0),
            "suspicious_tld_max": _bounded_float(summary.get("suspicious_tld_max"), upper=1.0),
            "uses_shortener_max": _bounded_float(summary.get("uses_shortener_max"), upper=1.0),
            "has_at_symbol_max": _bounded_float(summary.get("has_at_symbol_max"), upper=1.0),
            "subdomain_depth_mean": _bounded_float(summary.get("subdomain_depth_mean"), upper=50.0),
            "subdomain_depth_max": _bounded_float(summary.get("subdomain_depth_max"), upper=50.0),
            "path_depth_mean": _bounded_float(summary.get("path_depth_mean"), upper=100.0),
            "path_depth_max": _bounded_float(summary.get("path_depth_max"), upper=100.0),
            "num_special_chars_mean": _bounded_float(summary.get("num_special_chars_mean"), upper=100.0),
            "num_special_chars_max": _bounded_float(summary.get("num_special_chars_max"), upper=100.0),
            "num_url_params_mean": _bounded_float(summary.get("num_url_params_mean"), upper=100.0),
            "num_url_params_max": _bounded_float(summary.get("num_url_params_max"), upper=100.0),
            "url_digit_ratio_mean": _bounded_float(summary.get("url_digit_ratio_mean"), upper=1.0),
            "url_digit_ratio_max": _bounded_float(summary.get("url_digit_ratio_max"), upper=1.0),
        }
    )

    feature_dict.update(_status_one_hot("spf", signals.spf_status, ("pass", "fail", "softfail", "none")))
    feature_dict.update(_status_one_hot("dkim", signals.dkim_status, ("pass", "fail", "missing")))
    feature_dict.update(_status_one_hot("arc", signals.arc_status, ("pass", "fail", "missing")))
    return feature_dict


def _status_one_hot(prefix: str, raw_value: Any, known_values: tuple[str, ...]) -> dict[str, float]:
    """One-hot encode a status field with an explicit unknown bucket."""
    normalized = str(raw_value or "unknown").strip().lower()
    encoded = {f"{prefix}_{value}": 0.0 for value in (*known_values, "unknown")}
    if normalized not in known_values:
        normalized = "unknown"
    encoded[f"{prefix}_{normalized}"] = 1.0
    return encoded


def _bounded_float(value: Any, *, upper: float, allow_negative: bool = False) -> float:
    """Convert a value to a clipped non-negative float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return numeric if allow_negative else 0.0
    return min(numeric, upper)


def _clip_probability(value: Any) -> float:
    """Clamp arbitrary numeric model output into a valid probability."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "MetadataURLModel",
    "build_feature_dict_from_signals",
    "maybe_load_metadata_url_model",
]
