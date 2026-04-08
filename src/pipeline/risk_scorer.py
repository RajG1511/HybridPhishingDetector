"""Pipeline risk scoring utilities.

The risk scorer combines protocol, URL, and semantic signals into a single
0-100 phishing risk score using configurable layer weights from settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from config.settings import (
    LAYER1_MAX_RISK_POINTS,
    LAYER2_MAX_RISK_POINTS,
    LAYER3_MAX_RISK_POINTS,
    NEW_DOMAIN_THRESHOLD_DAYS,
    PHISHING_RISK_THRESHOLD,
    SAFE_RISK_THRESHOLD,
)
from src.pipeline.metadata_url_model import MetadataURLModel, maybe_load_metadata_url_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RiskSignals:
    """Container for the signals produced by Layers 1-3."""

    header_mismatch_count: int = 0
    dkim_valid: bool = True
    spf_pass: bool = True
    spf_status: str = "unknown"
    dkim_status: str = "unknown"
    arc_status: str = "unknown"
    protocol_risk_score: float | int | None = None
    metadata_flag_count: int = 0
    url_count: int = 0
    url_flags: list[str] = field(default_factory=list)
    url_feature_summary: Mapping[str, float] | None = None
    ensemble_proba: float = 0.0
    semantic_available: bool = True
    semantic_label: str = "legitimate"
    semantic_confidence: float | None = None
    semantic_probabilities: Mapping[str, float] | None = None


@dataclass(slots=True)
class RiskScoreResult:
    """Structured output from the risk scorer."""

    score: int
    label: str
    layer_scores: dict[str, int]
    metadata_url_probability: float | None = None
    metadata_url_model_name: str | None = None


class RiskScorer:
    """Compute weighted phishing risk from protocol, URL, and semantic signals."""

    def __init__(
        self,
        *,
        protocol_points: int = LAYER1_MAX_RISK_POINTS,
        url_points: int = LAYER2_MAX_RISK_POINTS,
        semantic_points: int = LAYER3_MAX_RISK_POINTS,
        safe_threshold: int = SAFE_RISK_THRESHOLD,
        phishing_threshold: int = PHISHING_RISK_THRESHOLD,
        metadata_url_model: MetadataURLModel | None = None,
        metadata_url_model_path: Path | str | None = None,
    ) -> None:
        self.protocol_points = protocol_points
        self.url_points = url_points
        self.semantic_points = semantic_points
        self.safe_threshold = safe_threshold
        self.phishing_threshold = phishing_threshold
        self.metadata_url_model = metadata_url_model
        if self.metadata_url_model is None and metadata_url_model_path is not None:
            self.metadata_url_model = maybe_load_metadata_url_model(metadata_url_model_path)

    def compute(self, signals: RiskSignals) -> int:
        """Return only the integer risk score."""
        return self.score(signals).score

    def score(self, signals: RiskSignals) -> RiskScoreResult:
        """Compute the final risk score and label.

        Args:
            signals: Aggregated Layer 1-3 signals.

        Returns:
            Structured score result with per-layer contributions.
        """
        (
            protocol_score,
            url_score,
            metadata_url_probability,
            metadata_url_model_name,
        ) = self._score_metadata_url(signals)
        layer_scores = {
            "layer1_protocol": protocol_score,
            "layer2_url": url_score,
            "layer3_semantic": self._score_semantic(signals),
        }
        
        # Dynamic Normalization: Calculate total possible points based on active layers
        active_max_points = self.protocol_points + self.url_points
        if signals.semantic_available:
            active_max_points += self.semantic_points
            
        raw_score_sum = sum(layer_scores.values())
        
        # Normalize to 0-100 scale regardless of how many layers are active
        final_score = round((raw_score_sum / active_max_points) * 100) if active_max_points > 0 else 0
        final_score = max(0, min(100, final_score))
        
        label = self.label_for_score(final_score)

        logger.debug(
            "Risk score=%d label=%s breakdown=%s",
            final_score,
            label,
            layer_scores,
        )
        return RiskScoreResult(
            score=final_score,
            label=label,
            layer_scores=layer_scores,
            metadata_url_probability=metadata_url_probability,
            metadata_url_model_name=metadata_url_model_name,
        )

    def label_for_score(self, score: int) -> str:
        """Map an integer risk score to a user-facing verdict label."""
        if score >= self.phishing_threshold:
            return "phishing"
        if score >= self.safe_threshold:
            return "suspicious"
        return "safe"

    def _score_metadata_url(
        self,
        signals: RiskSignals,
    ) -> tuple[int, int, float | None, str | None]:
        """Score Layers 1 and 2, optionally through a learned model."""
        rule_protocol_score = self._score_protocol_rule(signals)
        rule_url_score = self._score_url_rule(signals)

        if self.metadata_url_model is None:
            return rule_protocol_score, rule_url_score, None, None

        total_rule_points = rule_protocol_score + rule_url_score

        try:
            probability = self.metadata_url_model.predict_probability(signals)
        except Exception as exc:
            logger.warning("Metadata + URL model failed, falling back to rules: %s", exc)
            return rule_protocol_score, rule_url_score, None, None

        # Calculate pure ML points
        ml_points = round(probability * (self.protocol_points + self.url_points))
        
        # Hybrid Filter: Ensure the ML model can elevate the score, 
        # but cannot entirely wipe out hard protocol/URL rule violations.
        combined_points = min(
            self.protocol_points + self.url_points,
            max(total_rule_points, ml_points),
        )

        # Distribute points back proportionally based on rules, or evenly if no rules fired
        if total_rule_points > 0:
            protocol_score = round(combined_points * (rule_protocol_score / total_rule_points))
            # Cap protocol and shift excess to URL (up to its cap)
            protocol_score = min(protocol_score, self.protocol_points)
            url_score = min(combined_points - protocol_score, self.url_points)
        elif signals.url_count > 0:
            # If no rules fired but URLs exist, prioritize URL risk
            url_score = min(combined_points, self.url_points)
            protocol_score = min(combined_points - url_score, self.protocol_points)
        else:
            # Default fallback distribution relative to max possibilities
            protocol_score = round(combined_points * (self.protocol_points / (self.protocol_points + self.url_points)))
            # Final safety check on fallback distribution
            protocol_score = min(protocol_score, self.protocol_points)
            url_score = min(combined_points - protocol_score, self.url_points)

        return protocol_score, url_score, probability, self.metadata_url_model.model_name

    def _score_protocol_rule(self, signals: RiskSignals) -> int:
        """Score Layer 1 protocol evidence."""
        if signals.protocol_risk_score is not None:
            normalized = max(0.0, min(float(signals.protocol_risk_score), 100.0)) / 100.0
            return round(normalized * self.protocol_points)

        score = 0.0
        if not signals.spf_pass:
            score += 12
        if not signals.dkim_valid:
            score += 12

        arc_status = signals.arc_status.lower()
        if arc_status == "fail":
            score += 6
        elif arc_status == "missing":
            score += 3

        score += min(signals.header_mismatch_count * 3, 6)
        return min(self.protocol_points, round(score))

    def _score_url_rule(self, signals: RiskSignals) -> int:
        """Score Layer 2 URL evidence."""
        summary = signals.url_feature_summary or {}
        score = min(len(signals.url_flags) * 3, self.url_points)

        if summary:
            score += 6 * self._flag_value(summary, "has_ip_max")
            score += 6 * self._flag_value(summary, "has_homoglyph_max")
            score += 4 * self._flag_value(summary, "suspicious_tld_max")
            score += 3 * self._flag_value(summary, "uses_shortener_max")
            score += 3 * self._flag_value(summary, "has_at_symbol_max")

            if summary.get("url_digit_ratio_max", 0.0) >= 0.15:
                score += 2
            if summary.get("num_url_params_max", 0.0) >= 3:
                score += 1
            if summary.get("subdomain_depth_max", 0.0) >= 4:
                score += 2

        if signals.url_count == 0 and not summary and not signals.url_flags:
            return 0

        return min(self.url_points, round(score))

    def _score_semantic(self, signals: RiskSignals) -> int:
        """Score Layer 3 semantic evidence."""
        if not signals.semantic_available:
            return 0
        phishing_probability = self._phishing_probability(signals)
        return min(self.semantic_points, round(phishing_probability * self.semantic_points))

    @staticmethod
    def _flag_value(summary: Mapping[str, float], key: str) -> float:
        """Return a bounded summary feature value."""
        value = float(summary.get(key, 0.0))
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _phishing_probability(signals: RiskSignals) -> float:
        """Estimate the overall phishing probability from semantic outputs."""
        if signals.semantic_probabilities:
            phishing_probability = sum(
                probability
                for label, probability in signals.semantic_probabilities.items()
                if label != "legitimate"
            )
            return max(0.0, min(phishing_probability, 1.0))

        if signals.semantic_confidence is not None:
            confidence = max(0.0, min(float(signals.semantic_confidence), 1.0))
            if signals.semantic_label == "legitimate":
                return 1.0 - confidence
            return confidence

        return max(0.0, min(float(signals.ensemble_proba), 1.0))


__all__ = [
    "RiskScoreResult",
    "RiskScorer",
    "RiskSignals",
]
