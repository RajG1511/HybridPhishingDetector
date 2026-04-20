"""Demo runner for Layer 1 protocol authentication output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layer1_protocol import analyze_protocol_authentication
from src.pipeline.email_ingester import ingest_raw

PHISHING_SAMPLE = """\
From: "Security Team" <attacker@evil-bank.com>
Reply-To: collect@phish.net
Return-Path: <bounce@evil-bank.com>
Received: from mail.evil-bank.com (mail.evil-bank.com [203.0.113.10]) by mx.example with ESMTP
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
From: Alice Example <alice@company.com>
Reply-To: alice@company.com
Return-Path: <alice@company.com>
Message-ID: <q3-budget-2026@company.com>
Received-SPF: pass (domain of company.com designates 198.51.100.24 as permitted sender)
Authentication-Results: mx.company.com; spf=pass smtp.mailfrom=company.com
Received: from mail.company.com (mail.company.com [198.51.100.24]) by mx.company.com with ESMTP
Subject: Q3 Budget Review
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Hi Bob,

Please find attached the Q3 budget review document.

Best,
Alice
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Layer 1 demo."""
    parser = argparse.ArgumentParser(
        description="Run Layer 1 protocol authentication on a sample or .eml file."
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
        "--json",
        action="store_true",
        help="Print the full Layer 1 result as formatted JSON.",
    )
    return parser.parse_args()


def load_email_input(sample: str, file_path: Path | None) -> bytes:
    """Load demo email bytes from a sample or file path."""
    if file_path is not None:
        return file_path.read_bytes()
    if sample == "ham":
        return HAM_SAMPLE.encode("utf-8")
    return PHISHING_SAMPLE.encode("utf-8")


def build_demo_payload(raw_email: bytes) -> dict[str, Any]:
    """Build a readable Layer 1 demo payload."""
    parsed_email = ingest_raw(raw_email)
    layer1_output = analyze_protocol_authentication(parsed_email)

    return {
        "email_summary": {
            "from": parsed_email.from_addr,
            "reply_to": parsed_email.reply_to,
            "return_path": parsed_email.return_path,
            "subject": parsed_email.subject,
            "has_html_body": bool(parsed_email.html_body),
            "attachment_names": parsed_email.attachment_names,
        },
        "layer1": layer1_output,
    }


def print_human_readable(payload: dict[str, Any]) -> None:
    """Render Layer 1 output as a concise console summary."""
    email_summary = payload["email_summary"]
    layer1 = payload["layer1"]
    metadata_features = layer1["metadata_features"]

    print("=== Email Summary ===")
    print(f"From: {email_summary['from']}")
    print(f"Reply-To: {email_summary['reply_to']}")
    print(f"Return-Path: {email_summary['return_path']}")
    print(f"Subject: {email_summary['subject']}")
    print(f"Attachments: {email_summary['attachment_names']}")
    print()

    print("=== Layer 1 Status ===")
    print(f"SPF: {layer1['spf']}")
    print(f"DKIM: {layer1['dkim']}")
    print(f"ARC: {layer1['arc']}")
    print(f"Header mismatch: {layer1['header_mismatch']}")
    print(f"Protocol risk score: {layer1['protocol_risk_score']}")
    print()

    print("=== Header Issues ===")
    if layer1["header_issues"]:
        for issue in layer1["header_issues"]:
            print(f"- {issue}")
    else:
        print("None")
    print()

    print("=== Metadata Flags ===")
    if layer1["metadata_flags"]:
        for flag in layer1["metadata_flags"]:
            print(f"- {flag}")
    else:
        print("None")
    print()

    print("=== Metadata Features ===")
    for key in (
        "from_domain",
        "reply_to_domain",
        "return_path_domain",
        "message_id_domain",
        "sender_ip",
        "from_reply_to_mismatch",
        "from_return_path_mismatch",
        "message_id_domain_mismatch",
        "num_received_headers",
    ):
        print(f"{key}: {metadata_features.get(key)}")


def main() -> int:
    """Run the demo CLI."""
    args = parse_args()
    raw_email = load_email_input(sample=args.sample, file_path=args.file)
    payload = build_demo_payload(raw_email)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_human_readable(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
