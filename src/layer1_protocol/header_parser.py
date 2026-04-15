"""Layer 1 header parsing utilities."""

from __future__ import annotations

import email
import logging
from email import policy
from email.message import Message
from email.utils import parseaddr
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
        return email.message_from_bytes(raw, policy=policy.default)
    if isinstance(source, bytes):
        return email.message_from_bytes(source, policy=policy.default)
    return email.message_from_string(source, policy=policy.default)


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
        "authentication_results": _coerce_header_value(msg.get("Authentication-Results")),
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

    from_domain = extract_address_domain(fields.get("from"))
    reply_domain = extract_address_domain(fields.get("reply_to"))
    return_domain = extract_address_domain(fields.get("return_path"))

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


def extract_address_domain(value: str | None) -> str:
    """Extract the domain from a mailbox-style header value."""
    if not value:
        return ""
    _, parsed_address = parseaddr(value)
    if "@" not in parsed_address:
        return ""
    return parsed_address.rsplit("@", 1)[-1].strip(">").lower()


__all__ = [
    "detect_header_mismatches",
    "extract_address_domain",
    "extract_header_fields",
    "parse_eml",
]
