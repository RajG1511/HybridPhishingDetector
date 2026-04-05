"""
XAI Module — Narrative Explanation Generator

Converts raw SHAP/LIME feature attributions into human-readable natural
language explanations suitable for display in a security dashboard or email
client. Optionally calls an LLM for richer narrative generation.

Dependencies: None (LLM call is optional)
"""

import logging
import os

logger = logging.getLogger(__name__)


def generate_rule_based_narrative(
    risk_score: float,
    top_shap_features: list[tuple[str, float]],
    lime_words: list[tuple[str, float]],
    header_mismatches: list[str],
    url_flags: list[str],
) -> str:
    """Generate a human-readable explanation using rule-based templates.

    Args:
        risk_score: Final risk score in [0, 100].
        top_shap_features: Output of SHAPExplainer.top_features().
        lime_words: Output of LIMETextExplainer.explain_instance().
        header_mismatches: List of header mismatch descriptions from Layer 1.
        url_flags: List of URL risk flags from Layer 2.

    Returns:
        Multi-sentence plain-English explanation string.
    """
    verdict = "phishing" if risk_score >= 75 else ("suspicious" if risk_score >= 40 else "legitimate")
    parts: list[str] = [
        f"This email has been classified as **{verdict}** with a risk score of {risk_score:.0f}/100."
    ]

    if header_mismatches:
        parts.append(
            "Protocol checks raised the following concerns: " + "; ".join(header_mismatches) + "."
        )

    if url_flags:
        parts.append("URL analysis detected: " + "; ".join(url_flags) + ".")

    if lime_words:
        top_words = [w for w, s in lime_words[:5] if s > 0]
        if top_words:
            parts.append(
                "The following words contributed most to the phishing classification: "
                + ", ".join(f'"{w}"' for w in top_words) + "."
            )

    if top_shap_features:
        feat_str = ", ".join(f"{n} ({v:.3f})" for n, v in top_shap_features[:5])
        parts.append(f"Top model features driving this decision: {feat_str}.")

    return " ".join(parts)


def generate_llm_narrative(
    risk_score: float,
    rule_based_narrative: str,
) -> str:
    """Optionally enhance the rule-based narrative with an LLM call.

    Falls back gracefully to the rule-based narrative if the LLM endpoint
    is not configured or the call fails.

    Args:
        risk_score: Final risk score in [0, 100].
        rule_based_narrative: Output of generate_rule_based_narrative().

    Returns:
        Enhanced natural language explanation string.
    """
    import requests

    endpoint = os.getenv("LLM_API_ENDPOINT", "")
    model = os.getenv("LLM_MODEL_NAME", "llama4-scout")

    if not endpoint:
        return rule_based_narrative

    prompt = (
        "Rewrite the following technical phishing email analysis as a clear, concise explanation "
        "for a non-technical user. Keep it under 3 sentences.\n\n"
        f"Analysis: {rule_based_narrative}"
    )

    try:
        resp = requests.post(
            endpoint,
            json={"model": model, "prompt": prompt, "max_tokens": 150},
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json().get("text", rule_based_narrative).strip()
    except Exception as exc:
        logger.warning("LLM narrative generation failed: %s", exc)
        return rule_based_narrative
