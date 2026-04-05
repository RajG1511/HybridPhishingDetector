"""
Cascade Pipeline — Full Detection Orchestration

Executes the four-layer detection cascade:
  1. Protocol authentication (Layer 1)
  2. URL feature analysis (Layer 2) — if links present
  3. Semantic ensemble classification (Layer 3)
  4. RAG contextual profiling (Layer 4) — only for grey-zone scores

Returns a final risk score (0–100) and an XAI explanation object.
"""

import logging
from dataclasses import dataclass, field

from config.settings import GREY_ZONE_HIGH, GREY_ZONE_LOW
from src.pipeline.email_ingester import ParsedEmail
from src.pipeline.risk_scorer import RiskScorer, RiskSignals

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Output of the full cascade pipeline.

    Attributes:
        risk_score: Integer risk score 0–100.
        verdict: "benign", "suspicious", or "phishing".
        layer1_flags: Protocol authentication issues.
        layer2_flags: URL analysis flags.
        layer3_proba: Ensemble phishing probability (0.0–1.0).
        layer4_used: Whether the RAG layer was invoked.
        layer4_decision: RAG layer final decision (if invoked).
        narrative: Human-readable XAI explanation.
        shap_features: Top SHAP feature attributions.
        lime_words: Top LIME word attributions.
    """

    risk_score: int
    verdict: str
    layer1_flags: list[str] = field(default_factory=list)
    layer2_flags: list[str] = field(default_factory=list)
    layer3_proba: float = 0.0
    layer4_used: bool = False
    layer4_decision: str | None = None
    narrative: str = ""
    shap_features: list[tuple[str, float]] = field(default_factory=list)
    lime_words: list[tuple[str, float]] = field(default_factory=list)


class CascadePipeline:
    """Orchestrates the four-layer phishing detection cascade.

    Args:
        layer1: Initialized Layer 1 module (protocol auth). Pass None to skip.
        layer2: Initialized Layer 2 module (URL features). Pass None to skip.
        layer3: Fitted Layer 3 ensemble model. Pass None to skip.
        layer4: Initialized Layer 4 RAG engine. Pass None to skip.
        shap_explainer: Fitted SHAPExplainer. Pass None to skip.
        lime_explainer: Initialized LIMETextExplainer. Pass None to skip.
        vectorizer: Fitted TF-IDF vectorizer for Layer 3 input.
    """

    def __init__(
        self,
        layer1=None,
        layer2=None,
        layer3=None,
        layer4=None,
        shap_explainer=None,
        lime_explainer=None,
        vectorizer=None,
    ) -> None:
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.shap_explainer = shap_explainer
        self.lime_explainer = lime_explainer
        self.vectorizer = vectorizer
        self.scorer = RiskScorer()

    def run(self, parsed_email: ParsedEmail) -> DetectionResult:
        """Run the full detection cascade on a parsed email.

        Args:
            parsed_email: Output of email_ingester.ingest_raw() or ingest_eml_file().

        Returns:
            DetectionResult with risk score, verdict, and XAI explanation.
        """
        signals = RiskSignals()
        result = DetectionResult(risk_score=0, verdict="benign")

        # --- Layer 1: Protocol Authentication ---
        if self.layer1 is not None:
            try:
                from src.layer1_protocol.header_parser import detect_header_mismatches, extract_header_fields
                from src.layer1_protocol.dkim_verifier import verify_dkim, has_dkim_header
                from src.layer1_protocol.spf_checker import parse_received_spf_header, SPFResult

                fields = extract_header_fields(parsed_email.raw_bytes.decode("utf-8", errors="replace") if isinstance(parsed_email.raw_bytes, bytes) else parsed_email.raw_bytes)
                mismatches = detect_header_mismatches(fields)
                result.layer1_flags.extend(mismatches)

                dkim_valid = verify_dkim(parsed_email.raw_bytes)
                if not dkim_valid:
                    result.layer1_flags.append("DKIM verification failed or absent")

                spf_header = fields.get("received_spf") or ""
                spf_result = parse_received_spf_header(spf_header)
                if spf_result in (SPFResult.FAIL, SPFResult.SOFTFAIL):
                    result.layer1_flags.append(f"SPF result: {spf_result.value}")

                signals.header_mismatch_count = len(mismatches)
                signals.dkim_valid = dkim_valid
                signals.spf_pass = spf_result == SPFResult.PASS
            except Exception as exc:
                logger.warning("Layer 1 failed: %s", exc)

        # --- Layer 2: URL Feature Analysis ---
        body = parsed_email.html_body or parsed_email.plain_body
        if body:
            try:
                from src.layer2_url.url_extractor import extract_urls
                from src.layer2_url.lexical_features import extract_lexical_features

                urls = extract_urls(body)
                signals.url_count = len(urls)

                for url in urls[:20]:  # Limit to first 20 URLs
                    feats = extract_lexical_features(url)
                    if feats.get("has_ip_address"):
                        result.layer2_flags.append(f"IP-based URL: {url[:60]}")
                    if feats.get("has_homoglyphs"):
                        result.layer2_flags.append(f"Homoglyph characters in URL: {url[:60]}")
                    signals.url_flags.extend(result.layer2_flags)
            except Exception as exc:
                logger.warning("Layer 2 failed: %s", exc)

        # --- Layer 3: Semantic Ensemble ---
        if self.layer3 is not None and self.vectorizer is not None:
            try:
                import numpy as np
                from src.layer3_semantic.preprocessor import preprocess

                cleaned = preprocess(body or "")
                X = self.vectorizer.transform([cleaned]).toarray()
                proba = self.layer3.predict_proba(X)[0, 1]
                signals.ensemble_proba = proba
                result.layer3_proba = float(proba)
            except Exception as exc:
                logger.warning("Layer 3 failed: %s", exc)

        # --- Compute preliminary score ---
        preliminary_score = self.scorer.compute(signals)

        # --- Layer 4: RAG (grey zone only) ---
        if self.layer4 is not None and GREY_ZONE_LOW * 100 <= preliminary_score <= GREY_ZONE_HIGH * 100:
            try:
                import numpy as np
                from src.layer3_semantic.vectorizer import get_distilbert_embeddings

                embedding = get_distilbert_embeddings([body or ""])[0]
                rag_result = self.layer4.analyze(body or "", embedding)
                result.layer4_used = True
                result.layer4_decision = rag_result.get("decision")
                signals.rag_decision = result.layer4_decision
                signals.rag_confidence = float(rag_result.get("confidence", 0.5))
            except Exception as exc:
                logger.warning("Layer 4 failed: %s", exc)

        final_score = self.scorer.compute(signals)
        result.risk_score = final_score
        result.verdict = _score_to_verdict(final_score)

        logger.info("Pipeline complete — score=%d verdict=%s", final_score, result.verdict)
        return result


def _score_to_verdict(score: int) -> str:
    if score >= 75:
        return "phishing"
    if score >= 40:
        return "suspicious"
    return "benign"
