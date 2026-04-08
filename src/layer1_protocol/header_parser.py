"""
Layer 1 — Email Header Parser

Parses raw .eml files to extract and compare the From, Reply-To, and
Return-Path header fields. Flags mismatches as indicators of domain spoofing.

Dependencies: Python standard library `email` module.
"""

import email
import logging
from email.message import Message
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_eml(source: str | bytes | Path) -> Message:
    """Parse a raw .eml file or bytes into an email.message.Message object.

    Args:
        source: Path to a .eml file, raw email string, or raw email bytes.

    Returns:
        Parsed email Message object.
    """
    if isinstance(source, Path):
        raw = source.read_bytes()
        return email.message_from_bytes(raw)
    if isinstance(source, bytes):
        return email.message_from_bytes(source)
    return email.message_from_string(source)


def extract_header_fields(msg: Message) -> dict[str, str | None]:
    """Extract key authentication-relevant header fields.

    Args:
        msg: Parsed email Message object.

    Returns:
        Dictionary with keys: from, reply_to, return_path, message_id,
        received_spf, dkim_signature, arc_seal.
    """
    return {
        "from": _coerce_header_value(msg.get("From")),
        "reply_to": _coerce_header_value(msg.get("Reply-To")),
        "return_path": _coerce_header_value(msg.get("Return-Path")),
        "message_id": _coerce_header_value(msg.get("Message-ID")),
        "received_spf": _coerce_header_value(msg.get("Received-SPF")),
        "dkim_signature": _coerce_header_value(msg.get("DKIM-Signature")),
        "arc_seal": _coerce_header_value(msg.get("ARC-Seal")),
    }


def detect_header_mismatches(fields: dict[str, str | None]) -> list[str]:
    """Flag suspicious discrepancies between header fields.

    Args:
        fields: Output of extract_header_fields().

    Returns:
        List of human-readable mismatch descriptions (empty if none detected).
    """
    mismatches: list[str] = []

    from_addr = fields.get("from") or ""
    reply_to = fields.get("reply_to") or ""
    return_path = fields.get("return_path") or ""

    def _extract_domain(addr: str) -> str:
        if "@" in addr:
            return addr.split("@")[-1].strip(">").lower()
        return ""

    from_domain = _extract_domain(from_addr)
    reply_domain = _extract_domain(reply_to)
    return_domain = _extract_domain(return_path)

    if reply_domain and from_domain and reply_domain != from_domain:
        mismatches.append(
            f"Reply-To domain ({reply_domain}) differs from From domain ({from_domain})"
        )

    if return_domain and from_domain and return_domain != from_domain:
        mismatches.append(
            f"Return-Path domain ({return_domain}) differs from From domain ({from_domain})"
        )

    logger.debug("Header mismatch check: %d issues found", len(mismatches))
    return mismatches


def _coerce_header_value(value: object | None) -> str | None:
    """Normalize parsed header values into plain strings."""
    if value is None:
        return None
    return str(value)
