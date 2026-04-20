"""Layer 1 metadata feature extraction for raw emails.

This module converts raw email headers into structural phishing signals such as
sender mismatches, authentication status, and header-presence indicators.
"""

from __future__ import annotations

import email
import importlib.util
import logging
import re
from dataclasses import asdict, dataclass, field
from email.message import Message
from typing import Any

from src.layer1_protocol.arc_validator import validate_arc
from src.layer1_protocol.dkim_verifier import has_dkim_header, verify_dkim
from src.layer1_protocol.header_parser import (
    detect_header_mismatches,
    extract_address_domain,
    extract_header_fields,
)
from src.layer1_protocol.spf_checker import (
    SPFResult,
    extract_sender_ip,
    resolve_spf_status,
)

logger = logging.getLogger(__name__)

_MESSAGE_ID_DOMAIN_PATTERN = re.compile(r"@([^>]+)>?$")


@dataclass(slots=True)
class MetadataFeatureSet:
    """Structured metadata features derived from email headers."""

    from_address: str
    from_domain: str
    reply_to_address: str | None
    reply_to_domain: str
    return_path_address: str | None
    return_path_domain: str
    message_id: str | None
    message_id_domain: str
    sender_ip: str
    spf: str
    dkim: str
    arc: str
    has_reply_to: bool
    has_return_path: bool
    has_message_id: bool
    has_dkim_signature: bool
    has_received_spf: bool
    has_authentication_results: bool
    has_arc_headers: bool
    num_received_headers: int
    from_reply_to_mismatch: bool
    from_return_path_mismatch: bool
    message_id_domain_mismatch: bool
    header_mismatch: bool
    header_issues: list[str] = field(default_factory=list)
    metadata_flags: list[str] = field(default_factory=list)
    protocol_risk_score: int = 0

    def to_output_dict(self) -> dict[str, Any]:
        """Return a pipeline-friendly dictionary representation."""
        features = asdict(self)
        return {
            "spf": self.spf,
            "dkim": self.dkim,
            "arc": self.arc,
            "header_mismatch": self.header_mismatch,
            "header_issues": self.header_issues,
            "metadata_flags": self.metadata_flags,
            "protocol_risk_score": self.protocol_risk_score,
            "metadata_features": features,
        }


def extract_metadata_features(email_input: Message | bytes | str | Any) -> MetadataFeatureSet:
    """Extract metadata/authentication features from an email.

    Args:
        email_input: Raw email bytes/string, parsed Message, or ParsedEmail-like
            object with a ``raw_bytes`` attribute.

    Returns:
        Structured metadata feature set.
    """
    message, raw_bytes = _normalize_email_input(email_input)
    fields = extract_header_fields(message)
    received_headers = [_safe_string(value) for value in message.get_all("Received", [])]

    from_address = fields.get("from") or ""
    reply_to_address = fields.get("reply_to")
    return_path_address = fields.get("return_path")
    message_id = fields.get("message_id")

    from_domain = extract_address_domain(from_address)
    reply_to_domain = extract_address_domain(reply_to_address or "")
    return_path_domain = extract_address_domain(return_path_address or "")
    message_id_domain = _extract_message_id_domain(message_id)
    sender_ip = extract_sender_ip(received_headers) or ""

    header_issues = detect_header_mismatches(fields)
    from_reply_to_mismatch = bool(reply_to_domain and from_domain and reply_to_domain != from_domain)
    from_return_path_mismatch = bool(
        return_path_domain and from_domain and return_path_domain != from_domain
    )
    message_id_domain_mismatch = bool(
        message_id_domain and from_domain and message_id_domain != from_domain
    )
    if message_id_domain_mismatch:
        header_issues.append(
            f"Message-ID domain ({message_id_domain}) differs from From domain ({from_domain})"
        )

    spf_status = resolve_spf_status(
        domain=return_path_domain or from_domain,
        sender_ip=sender_ip or None,
        received_spf_header=fields.get("received_spf"),
        authentication_results_header=fields.get("authentication_results"),
    ).value
    dkim_status = _detect_dkim_status(raw_bytes, bool(fields.get("dkim_signature")))
    arc_status = _detect_arc_status(raw_bytes)

    metadata_flags = _build_metadata_flags(
        from_reply_to_mismatch=from_reply_to_mismatch,
        from_return_path_mismatch=from_return_path_mismatch,
        message_id_domain_mismatch=message_id_domain_mismatch,
        spf_status=spf_status,
        dkim_status=dkim_status,
        arc_status=arc_status,
        has_message_id=bool(message_id),
        has_authentication_results=bool(message.get("Authentication-Results")),
        num_received_headers=len(received_headers),
    )

    protocol_risk_score = _score_metadata_risk(
        from_reply_to_mismatch=from_reply_to_mismatch,
        from_return_path_mismatch=from_return_path_mismatch,
        message_id_domain_mismatch=message_id_domain_mismatch,
        spf_status=spf_status,
        dkim_status=dkim_status,
        arc_status=arc_status,
        has_message_id=bool(message_id),
        has_authentication_results=bool(message.get("Authentication-Results")),
        num_received_headers=len(received_headers),
    )

    return MetadataFeatureSet(
        from_address=from_address,
        from_domain=from_domain,
        reply_to_address=reply_to_address,
        reply_to_domain=reply_to_domain,
        return_path_address=return_path_address,
        return_path_domain=return_path_domain,
        message_id=message_id,
        message_id_domain=message_id_domain,
        sender_ip=sender_ip,
        spf=spf_status,
        dkim=dkim_status,
        arc=arc_status,
        has_reply_to=bool(reply_to_address),
        has_return_path=bool(return_path_address),
        has_message_id=bool(message_id),
        has_dkim_signature=bool(fields.get("dkim_signature")),
        has_received_spf=bool(fields.get("received_spf")),
        has_authentication_results=bool(message.get("Authentication-Results")),
        has_arc_headers=bool(fields.get("arc_seal")),
        num_received_headers=len(received_headers),
        from_reply_to_mismatch=from_reply_to_mismatch,
        from_return_path_mismatch=from_return_path_mismatch,
        message_id_domain_mismatch=message_id_domain_mismatch,
        header_mismatch=bool(header_issues),
        header_issues=header_issues,
        metadata_flags=metadata_flags,
        protocol_risk_score=protocol_risk_score,
    )


def _normalize_email_input(email_input: Message | bytes | str | Any) -> tuple[Message, bytes]:
    """Normalize different email inputs into a parsed Message plus raw bytes."""
    if isinstance(email_input, Message):
        raw_bytes = email_input.as_bytes()
        return email_input, raw_bytes

    if isinstance(email_input, bytes):
        return email.message_from_bytes(email_input), email_input

    if isinstance(email_input, str):
        raw_bytes = email_input.encode("utf-8", errors="replace")
        return email.message_from_string(email_input), raw_bytes

    raw_bytes = getattr(email_input, "raw_bytes", None)
    if isinstance(raw_bytes, bytes):
        return email.message_from_bytes(raw_bytes), raw_bytes

    raise TypeError("Unsupported email input type for metadata feature extraction")


def _extract_message_id_domain(message_id: str | None) -> str:
    """Extract the right-hand-side domain-like token from a Message-ID."""
    if not message_id:
        return ""
    match = _MESSAGE_ID_DOMAIN_PATTERN.search(message_id.strip())
    return match.group(1).lower() if match else ""


def _detect_dkim_status(raw_bytes: bytes, has_signature: bool) -> str:
    """Return pass/fail/missing/unknown DKIM status."""
    if not has_signature and not has_dkim_header(raw_bytes):
        return "missing"
    if importlib.util.find_spec("dkim") is None:
        return "unknown"
    return "pass" if verify_dkim(raw_bytes) else "fail"


def _detect_arc_status(raw_bytes: bytes) -> str:
    """Return pass/fail/missing/unknown ARC status."""
    result = validate_arc(raw_bytes)
    if not result.get("has_arc"):
        return "missing"
    if result.get("chain_valid") is True:
        return "pass"
    if result.get("chain_valid") is False:
        return "fail"
    cv = str(result.get("cv") or "").lower()
    if cv in {"pass", "fail"}:
        return cv
    return "unknown"


def analyze_protocol_authentication(email_input: Message | bytes | str | Any) -> dict[str, Any]:
    """Run Layer 1 analysis and return the pipeline contract dictionary."""
    return extract_metadata_features(email_input).to_output_dict()


def _safe_string(value: object | None) -> str:
    if value is None:
        return ""
    return str(value)


def _build_metadata_flags(
    *,
    from_reply_to_mismatch: bool,
    from_return_path_mismatch: bool,
    message_id_domain_mismatch: bool,
    spf_status: str,
    dkim_status: str,
    arc_status: str,
    has_message_id: bool,
    has_authentication_results: bool,
    num_received_headers: int,
) -> list[str]:
    """Build human-readable metadata flags from extracted features."""
    flags: list[str] = []

    if from_reply_to_mismatch:
        flags.append("Reply-To domain mismatch")
    if from_return_path_mismatch:
        flags.append("Return-Path domain mismatch")
    if message_id_domain_mismatch:
        flags.append("Message-ID domain mismatch")

    if spf_status == SPFResult.FAIL.value:
        flags.append("SPF failure")
    elif spf_status == SPFResult.SOFTFAIL.value:
        flags.append("SPF softfail")
    elif spf_status == SPFResult.NONE.value:
        flags.append("Missing SPF evidence")

    if dkim_status == "fail":
        flags.append("DKIM verification failed")
    elif dkim_status == "missing":
        flags.append("Missing DKIM signature")

    if arc_status == "fail":
        flags.append("ARC validation failed")
    elif arc_status == "missing":
        flags.append("Missing ARC headers")

    if not has_message_id:
        flags.append("Missing Message-ID header")
    if not has_authentication_results:
        flags.append("Missing Authentication-Results header")
    if num_received_headers == 0:
        flags.append("Missing Received headers")

    return flags


def _score_metadata_risk(
    *,
    from_reply_to_mismatch: bool,
    from_return_path_mismatch: bool,
    message_id_domain_mismatch: bool,
    spf_status: str,
    dkim_status: str,
    arc_status: str,
    has_message_id: bool,
    has_authentication_results: bool,
    num_received_headers: int,
) -> int:
    """Score metadata/header risk on a 0-100 scale."""
    score = 0

    if from_reply_to_mismatch:
        score += 20
    if from_return_path_mismatch:
        score += 20
    if message_id_domain_mismatch:
        score += 10

    if spf_status == SPFResult.FAIL.value:
        score += 20
    elif spf_status == SPFResult.SOFTFAIL.value:
        score += 12
    elif spf_status == SPFResult.NONE.value:
        score += 6

    if dkim_status == "fail":
        score += 20
    elif dkim_status == "missing":
        score += 8

    if arc_status == "fail":
        score += 8
    elif arc_status == "missing":
        score += 4

    if not has_message_id:
        score += 4
    if not has_authentication_results:
        score += 4
    if num_received_headers == 0:
        score += 4

    return min(score, 100)


__all__ = [
    "MetadataFeatureSet",
    "analyze_protocol_authentication",
    "extract_metadata_features",
]
