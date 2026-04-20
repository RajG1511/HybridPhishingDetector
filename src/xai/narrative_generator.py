"""
XAI Module — Narrative Explanation Generator

Converts LIME and SHAP outputs into a human-readable plain-English explanation
using rule-based templates. No LLM required — can be swapped in later.

Public API:
    generate_explanation(lime_result, shap_result, header_flags, url_flags) -> str
    format_explanation_block(email_id, lime_result, shap_result, ...) -> str
"""

import logging

logger = logging.getLogger(__name__)

# ── Label display strings ──────────────────────────────────────────────────
_LABEL_DISPLAY = {
    "legitimate":     "LEGITIMATE",
    "phishing_human": "HUMAN-WRITTEN PHISHING",
    "phishing_ai":    "AI-GENERATED PHISHING",
}

# ── Words strongly associated with each class (for narrative enrichment) ──
_CLASS_SIGNALS = {
    "phishing_human": [
        "verify", "urgent", "immediately", "suspend", "account", "password",
        "click", "confirm", "credentials", "update", "security", "alert",
    ],
    "phishing_ai": [
        "verify", "immediately", "credentials", "security", "team", "official",
        "legitimate", "ensure", "protect", "hereby", "inform", "compliance",
    ],
    "legitimate": [
        "meeting", "project", "attached", "regards", "team", "update",
        "please", "schedule", "review", "draft", "feedback",
    ],
}

# ── Attack-pattern descriptions keyed to trigger words ───────────────────
_ATTACK_PATTERNS = {
    "verify":      "account verification lure",
    "credential":  "credential harvesting attempt",
    "urgent":      "urgency manipulation tactic",
    "immediately": "urgency manipulation tactic",
    "suspend":     "account suspension threat",
    "password":    "password harvesting attempt",
    "click":       "click-through redirection lure",
    "security":    "security alarm trigger",
    "wire":        "financial fraud signal",
    "transfer":    "financial fraud signal",
    "invoice":     "invoice fraud signal",
    "prince":      "advance-fee fraud (Nigerian 419) pattern",
    "million":     "advance-fee fraud (Nigerian 419) pattern",
    "lottery":     "prize/lottery scam pattern",
    "won":         "prize/lottery scam pattern",
    "inheritance": "advance-fee fraud pattern",
}


def _get_attack_pattern(word: str) -> str | None:
    """Return a human-readable attack pattern for a known phishing keyword."""
    return _ATTACK_PATTERNS.get(word.lower())


def generate_explanation(
    lime_result: dict,
    shap_result: dict | None = None,
    header_flags: list[str] | None = None,
    url_flags: list[str] | None = None,
    risk_score: int | None = None,
    verdict: str | None = None,
) -> str:
    """Generate a human-readable explanation from LIME, SHAP, and multi-layer signals.

    Args:
        lime_result: Output of LIMEExplainer.explain_prediction().
        shap_result: Output of SHAPExplainer.explain_local() (optional).
        header_flags: Layer 1 protocol flags (optional).
        url_flags: Layer 2 URL risk flags (optional).
        risk_score: Final ensemble risk score (optional).
        verdict: Final ensemble verdict (optional).

    Returns:
        Multi-sentence plain-English explanation string.
    """
    label       = lime_result.get("predicted_label", "unknown")
    confidence  = lime_result.get("confidence", 0.0)
    features    = lime_result.get("top_features", [])
    class_probs = lime_result.get("class_probabilities", {})

    display = _LABEL_DISPLAY.get(label, label.upper())
    conf_pct = confidence * 100

    lines: list[str] = []

    # ── Verdict line ──────────────────────────────────────────────────────
    if verdict and risk_score is not None:
        lines.append(
            f"Analysis concludes this email is {verdict.upper()} with an overall risk score of {risk_score}/100."
        )
    else:
        lines.append(
            f"This email is predicted as {display} with {conf_pct:.1f}% semantic confidence."
        )

    # ── Semantic context ──────────────────────────────────────────────────
    if label == "legitimate" and (header_flags or url_flags):
        lines.append(
            "The message content appears authentic and safe based on linguisitic patterns."
        )
    elif label != "legitimate":
        lines.append(
            f"The semantic analysis flags the content as {display}."
        )

    # ── Tension & Reconciliation ──────────────────────────────────────────
    if label == "legitimate" and risk_score is not None and risk_score > 30:
        lines.append(
            f"However, technical metadata and security headers raise suspicious signals (Score: {risk_score}), "
            "suggesting this could be an impersonation attempt or a misconfigured sender."
        )

    # ── Probability breakdown ─────────────────────────────────────────────
    if class_probs:
        prob_parts = [
            f"{_LABEL_DISPLAY.get(k, k)}: {v*100:.1f}%"
            for k, v in sorted(class_probs.items(), key=lambda x: -x[1])
        ]
        lines.append(f"Semantic breakdown: {' | '.join(prob_parts)}.")

    # ── LIME top words ────────────────────────────────────────────────────
    if features:
        # Positive weights push toward predicted class; negative push away
        phishing_words = [(w, wt) for w, wt in features if wt > 0][:5]
        legit_words    = [(w, wt) for w, wt in features if wt < 0][:3]

        if phishing_words:
            word_phrases: list[str] = []
            seen_patterns: set[str] = set()
            for word, wt in phishing_words:
                pattern = _get_attack_pattern(word)
                phrase = f'"{word}" ({wt:+.3f})'
                if pattern and pattern not in seen_patterns:
                    phrase += f" [{pattern}]"
                    seen_patterns.add(pattern)
                word_phrases.append(phrase)
            
            if label != "legitimate":
                lines.append(f"Key suspicious indicators in text: {', '.join(word_phrases)}.")
            else:
                lines.append(f"Minor content anomalies detected: {', '.join(word_phrases)}.")

        if legit_words and label != "legitimate":
            counter_parts = [f'"{w}" ({wt:+.3f})' for w, wt in legit_words]
            lines.append(
                f"Mitigating authentic signals: {', '.join(counter_parts)}."
            )

    # ── SHAP context ─────────────────────────────────────────────────────
    if shap_result:
        shap_cls = shap_result.get("per_class", {}).get(label, [])
        if shap_cls:
            top_shap = shap_cls[:3]
            shap_parts = [f'"{w}" ({v:+.4f})' for w, v in top_shap]
            lines.append(
                f"Ensemble feature confirmation for {display}: {', '.join(shap_parts)}."
            )

    # ── Protocol / URL flags ──────────────────────────────────────────────
    if header_flags:
        lines.append(f"Security Header Alerts: {'; '.join(header_flags[:4])}.")
    if url_flags:
        lines.append(f"Suspicious URL attributes: {'; '.join(url_flags[:4])}.")

    # ── Conclusion ────────────────────────────────────────────────────────
    if verdict == "safe" or (verdict == "suspicious" and label == "legitimate"):
        lines.append(
            "Overall, the evidence suggests legitimate correspondence, though protocol warnings indicate "
            "minor configuration issues with the sender's email setup."
        )
    elif verdict == "phishing":
        lines.append(
            "The combination of high-risk metadata and suspicious content patterns indicates a high probability of phishing."
        )

    # ── Class-specific context ────────────────────────────────────────────
    if label == "phishing_ai":
        lines.append(
            "Note: AI-generated phishing emails often use polished, formal language "
            "that mimics legitimate communications — exercising extra caution is advised."
        )
    elif label == "phishing_human":
        lines.append(
            "This phishing email uses typical social engineering tactics to pressure "
            "the recipient into taking immediate action."
        )
    elif label == "legitimate":
        lines.append(
            "No significant phishing indicators were detected. "
            "The email content and structure are consistent with legitimate correspondence."
        )

    return " ".join(lines)


def format_explanation_block(
    email_id: str,
    raw_text_snippet: str,
    lime_result: dict,
    shap_result: dict | None = None,
    header_flags: list[str] | None = None,
    url_flags: list[str] | None = None,
    true_label: str | None = None,
) -> str:
    """Format a complete explanation block for display or logging.

    Args:
        email_id: Short identifier for the email (e.g. "test_email_001").
        raw_text_snippet: First ~200 characters of the email body.
        lime_result: Output of LIMEExplainer.explain_prediction().
        shap_result: Output of SHAPExplainer.explain_local() (optional).
        header_flags: Protocol authentication flags (optional).
        url_flags: URL risk flags (optional).
        true_label: Ground-truth label, if known (for demo/evaluation display).

    Returns:
        Formatted multi-line string ready for printing.
    """
    sep = "=" * 65
    pred  = lime_result.get("predicted_label", "?")
    conf  = lime_result.get("confidence", 0.0)
    truth = f" (True: {_LABEL_DISPLAY.get(true_label, true_label)})" if true_label else ""
    correct = " ✓" if true_label == pred else " ✗" if true_label else ""

    narrative = generate_explanation(lime_result, shap_result, header_flags, url_flags)

    lines = [
        sep,
        f"EMAIL: {email_id}{truth}{correct}",
        f"PREDICTED: {_LABEL_DISPLAY.get(pred, pred)} ({conf*100:.1f}% confidence)",
        sep,
        f'TEXT SNIPPET: "{raw_text_snippet[:200].strip()}..."',
        "",
        "EXPLANATION:",
        narrative,
        "",
        "TOP LIME FEATURES (word → weight toward predicted class):",
    ]

    for word, weight in (lime_result.get("top_features") or [])[:10]:
        direction = "→ phishing" if weight > 0 else "→ legitimate"
        lines.append(f"  {word:20s}  {weight:+.4f}  ({direction})")

    if shap_result:
        pred_cls_shap = shap_result.get("per_class", {}).get(pred, [])
        if pred_cls_shap:
            lines.append("")
            lines.append(f"TOP SHAP FEATURES (Random Forest, class={_LABEL_DISPLAY.get(pred, pred)}):")
            for word, val in pred_cls_shap[:10]:
                lines.append(f"  {word:20s}  {val:+.5f}")

    lines.append(sep)
    return "\n".join(lines)
