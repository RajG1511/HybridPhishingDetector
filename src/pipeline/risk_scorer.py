"""
Pipeline — Risk Scorer

Aggregates signals from all four detection layers into a final integer risk
score in the range [0, 100]. Higher scores indicate higher phishing likelihood.

Scoring weights are designed to reflect the relative discriminative power of
each signal source, with the ensemble probability carrying the most weight.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskSignals:
    """Container for all signals collected across pipeline layers.

    Attributes:
        header_mismatch_count: Number of header field mismatches (Layer 1).
        dkim_valid: Whether DKIM verification passed (Layer 1).
        spf_pass: Whether SPF check returned pass (Layer 1).
        url_count: Total URLs found in the email (Layer 2).
        url_flags: List of URL risk flag strings (Layer 2).
        ensemble_proba: Phishing probability from the Super Learner (Layer 3).
        rag_decision: "phishing" or "benign" from RAG layer (Layer 4).
        rag_confidence: RAG layer confidence score 0.0–1.0 (Layer 4).
    """

    header_mismatch_count: int = 0
    dkim_valid: bool = True
    spf_pass: bool = True
    url_count: int = 0
    url_flags: list[str] = field(default_factory=list)
    ensemble_proba: float = 0.0
    rag_decision: str | None = None
    rag_confidence: float = 0.5


class RiskScorer:
    """Computes the final risk score from aggregated detection signals.

    Weights:
        - Ensemble probability (Layer 3):    50 points max
        - DKIM failure (Layer 1):            15 points
        - SPF failure (Layer 1):             10 points
        - Header mismatches (Layer 1):        5 points each (max 10)
        - URL risk flags (Layer 2):           2 points each (max 10)
        - RAG decision override (Layer 4):   up to ±10 points adjustment
    """

    def compute(self, signals: RiskSignals) -> int:
        """Compute the risk score from the given signals.

        Args:
            signals: Populated RiskSignals dataclass.

        Returns:
            Integer risk score in [0, 100].
        """
        score = 0.0

        # Layer 3: Ensemble probability (0–50 points)
        score += signals.ensemble_proba * 50

        # Layer 1: Protocol failures
        if not signals.dkim_valid:
            score += 15
        if not signals.spf_pass:
            score += 10
        score += min(signals.header_mismatch_count * 5, 10)

        # Layer 2: URL flags
        score += min(len(signals.url_flags) * 2, 10)

        # Layer 4: RAG adjustment
        if signals.rag_decision is not None:
            rag_weight = (signals.rag_confidence - 0.5) * 20  # ±10 points
            if signals.rag_decision == "phishing":
                score += rag_weight
            else:
                score -= rag_weight

        final = max(0, min(100, round(score)))
        logger.debug(
            "Risk score: %d (ensemble=%.2f, dkim=%s, spf=%s, url_flags=%d, rag=%s)",
            final,
            signals.ensemble_proba,
            signals.dkim_valid,
            signals.spf_pass,
            len(signals.url_flags),
            signals.rag_decision,
        )
        return final
