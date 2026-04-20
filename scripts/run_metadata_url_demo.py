"""Demo runner for metadata and URL feature engineering work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import METADATA_URL_MODEL_PATH
from src.pipeline.cascade_pipeline import CascadePipeline
from src.pipeline.email_ingester import ingest_raw
from src.pipeline.risk_scorer import RiskScorer, RiskSignals

PHISHING_SAMPLE = """\
From: attacker@evil-bank.com
Reply-To: collect@phish.net
Return-Path: <bounce@evil-bank.com>
Subject: Urgent: Verify your account
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Dear Customer,

Your account has been suspended. Click here immediately to verify:
http://192.168.1.1/login?redirect=http://evil-bank.com/steal

Regards,
Security Team
"""

HAM_SAMPLE = """\
From: alice@company.com
Reply-To: alice@company.com
Return-Path: <alice@company.com>
Message-ID: <q3-budget-2026@company.com>
Received-SPF: pass (domain of company.com designates 203.0.113.10 as permitted sender)
Authentication-Results: mx.company.com; spf=pass smtp.mailfrom=company.com
Received: from mail.company.com by mx.company.com
Subject: Q3 Budget Review
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Hi Bob,

Please find attached the Q3 budget review document.

Best,
Alice
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo runner."""
    parser = argparse.ArgumentParser(
        description="Run a metadata + URL feature engineering demo on a sample or .eml file."
    )
    parser.add_argument(
        "--sample",
        choices=("phishing", "ham"),
        default="phishing",
        help="Use a built-in sample email.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Optional path to a raw .eml file. Overrides --sample.",
    )
    parser.add_argument(
        "--show-pipeline",
        action="store_true",
        help="Also run the current cascade pipeline and show the integrated output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the demo payload as formatted JSON.",
    )
    return parser.parse_args()


def load_email_input(sample: str, file_path: Path | None) -> bytes:
    """Load demo email bytes from a sample or file path."""
    if file_path is not None:
        return file_path.read_bytes()
    if sample == "ham":
        return HAM_SAMPLE.encode("utf-8")
    return PHISHING_SAMPLE.encode("utf-8")


def build_demo_payload(raw_email: bytes, show_pipeline: bool) -> dict[str, Any]:
    """Assemble metadata, URL, and optional pipeline output for a demo email."""
    parsed_email = ingest_raw(raw_email)
    pipeline = CascadePipeline()
    layer1_output = pipeline._run_layer1(parsed_email)
    layer2_output = pipeline._run_layer2(parsed_email)
    metadata_url_risk = _score_metadata_and_url(layer1_output, layer2_output)

    payload: dict[str, Any] = {
        "email_summary": {
            "from": parsed_email.from_addr,
            "subject": parsed_email.subject,
            "body_has_html": bool(parsed_email.html_body),
            "url_count": int(layer2_output.get("url_count", 0)),
        },
        "metadata": layer1_output,
        "urls": list(layer2_output.get("urls", [])),
        "url_feature_summary": dict(layer2_output.get("feature_summary", {})),
        "per_url_features": list(layer2_output.get("per_url_features", [])),
        "domain_intel": list(layer2_output.get("domain_intel", [])),
        "metadata_url_risk": metadata_url_risk,
    }

    if show_pipeline:
        result = pipeline.run(parsed_email)
        payload["pipeline"] = {
            "risk_score": result.risk_score,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "layer_outputs": result.layer_outputs,
            "layer4_used": result.layer4_used,
            "explanation": result.explanation,
        }

    return payload


def print_human_readable(payload: dict[str, Any]) -> None:
    """Render the demo payload as a readable console walkthrough."""
    email_summary = payload["email_summary"]
    metadata = payload["metadata"]

    print("=== Email Summary ===")
    print(f"From: {email_summary['from']}")
    print(f"Subject: {email_summary['subject']}")
    print(f"URLs found: {email_summary['url_count']}")
    print()

    print("=== Metadata Signals ===")
    print(f"SPF: {metadata['spf']}")
    print(f"DKIM: {metadata['dkim']}")
    print(f"ARC: {metadata['arc']}")
    print(f"Protocol risk score: {metadata['protocol_risk_score']}")
    print(f"Header issues: {metadata['header_issues']}")
    print(f"Metadata flags: {metadata['metadata_flags']}")
    print()

    print("=== URLs ===")
    if payload["urls"]:
        for index, url in enumerate(payload["urls"], start=1):
            print(f"{index}. {url}")
    else:
        print("No URLs found in the email body.")
    print()

    print("=== URL Summary Features ===")
    for key in (
        "url_count",
        "has_ip_max",
        "has_homoglyph_max",
        "suspicious_tld_max",
        "subdomain_depth_max",
        "url_entropy_max",
    ):
        print(f"{key}: {payload['url_feature_summary'].get(key)}")
    print()

    if payload["domain_intel"]:
        print("=== Domain Intelligence ===")
        for item in payload["domain_intel"]:
            print(
                f"{item['domain']}: age_days={item['domain_age_days']}, "
                f"is_new_domain={item['is_new_domain']}, status={item['lookup_status']}"
            )
    print()

    print("=== Metadata + URL Risk Breakdown ===")
    person2_risk = payload["metadata_url_risk"]
    person2_layer_scores = person2_risk["layer_scores"]
    layer_maxes = person2_risk["layer_maxes"]
    print(
        f"Layer 1 metadata/protocol: "
        f"{person2_layer_scores['layer1_protocol']} / {layer_maxes['layer1_protocol']}"
    )
    print(
        f"Layer 2 URL: "
        f"{person2_layer_scores['layer2_url']} / {layer_maxes['layer2_url']}"
    )
    print(
        f"Metadata + URL subtotal: "
        f"{person2_risk['subtotal']} / {person2_risk['subtotal_max']}"
    )
    print(
        f"Metadata + URL normalized to 100: "
        f"{person2_risk['normalized_to_100']} / 100"
    )
    print(f"Metadata + URL threshold: {person2_risk['threshold']} / 100")
    print(f"Metadata + URL verdict: {person2_risk['normalized_label']}")
    if person2_risk.get("model_name"):
        print(
            f"Metadata + URL model: {person2_risk['model_name']} "
            f"(phishing_probability={float(person2_risk.get('probability') or 0.0):.2f})"
        )
    else:
        print("Metadata + URL model: rule-based fallback")
    print()

    if "pipeline" in payload:
        pipeline = payload["pipeline"]
        print("=== Pipeline Output ===")
        full_layer_scores = pipeline["layer_outputs"]["risk"]["layer_scores"]
        print(f"Layer 1 metadata/protocol: {full_layer_scores['layer1_protocol']} / 30")
        print(f"Layer 2 URL: {full_layer_scores['layer2_url']} / 25")
        print(f"Layer 3 semantic: {full_layer_scores['layer3_semantic']} / 45")
        if not pipeline["layer_outputs"]["layer3"].get("model_loaded", False):
            print("Layer 3 note: semantic model not loaded, so this contributes 0 points.")
            print("Pipeline note: this full-project score is partial; use the metadata + URL verdict above for this demo.")
        layer4 = pipeline["layer_outputs"]["layer4"]
        print(
            "Layer 4 note: "
            f"{layer4['note']} "
            f"(eligible={layer4['eligible']}, used={pipeline['layer4_used']})"
        )
        print(f"Full pipeline risk score: {pipeline['risk_score']} / 100")
        print(f"Predicted label: {pipeline['predicted_label']}")
        print(f"Confidence: {pipeline['confidence']:.2f}")
        print(f"Explanation: {pipeline['explanation']}")


def _score_metadata_and_url(
    layer1_output: dict[str, Any],
    layer2_output: dict[str, Any],
) -> dict[str, Any]:
    """Score only the metadata and URL layers for demo purposes."""
    scorer = RiskScorer(metadata_url_model_path=METADATA_URL_MODEL_PATH)
    signals = RiskSignals(
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
        semantic_available=False,
    )
    result = scorer.score(signals)
    subtotal = result.layer_scores["layer1_protocol"] + result.layer_scores["layer2_url"]
    subtotal_max = scorer.protocol_points + scorer.url_points
    normalized_to_100 = round((subtotal / subtotal_max) * 100) if subtotal_max else 0
    metadata_url_threshold = (
        round(float(scorer.metadata_url_model.threshold) * 100)
        if scorer.metadata_url_model is not None
        else scorer.safe_threshold
    )
    normalized_label = _metadata_url_label_for_score(
        normalized_to_100,
        threshold=metadata_url_threshold,
        scorer=scorer,
    )

    return {
        "score": result.score,
        "layer_scores": result.layer_scores,
        "subtotal": subtotal,
        "subtotal_max": subtotal_max,
        "normalized_to_100": normalized_to_100,
        "normalized_label": normalized_label,
        "threshold": metadata_url_threshold,
        "probability": result.metadata_url_probability,
        "model_name": result.metadata_url_model_name,
        "layer_maxes": {
            "layer1_protocol": scorer.protocol_points,
            "layer2_url": scorer.url_points,
            "layer3_semantic": scorer.semantic_points,
        },
    }


def _metadata_url_label_for_score(
    score: float,
    *,
    threshold: float,
    scorer: RiskScorer,
) -> str:
    """Return a metadata+URL verdict using the active scoring regime."""
    if scorer.metadata_url_model is not None:
        return "phishing" if score >= threshold else "safe"
    return scorer.label_for_score(score)


def main() -> None:
    """Run the metadata + URL feature engineering demo."""
    args = parse_args()
    raw_email = load_email_input(args.sample, args.file)
    payload = build_demo_payload(raw_email, show_pipeline=args.show_pipeline)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    print_human_readable(payload)


if __name__ == "__main__":
    main()
