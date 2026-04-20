"""
Pipeline — Email Ingester

Parses .eml files and raw email strings/bytes into a normalized internal
representation consumed by the cascade pipeline. Extracts headers, plain-text
body, HTML body, attachments, and raw bytes for protocol-level checks.

Dependencies: Python standard library `email` module, beautifulsoup4
"""

import email
import logging
import re
from dataclasses import dataclass, field
from email import policy
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedEmail:
    """Normalized internal representation of an email.

    Attributes:
        raw_bytes: Original raw email bytes.
        headers: Dict of all header fields.
        from_addr: Sender address string.
        reply_to: Reply-To address string (may be None).
        return_path: Return-Path address string (may be None).
        subject: Email subject line.
        plain_body: Plain-text body content.
        html_body: HTML body content (may be empty).
        attachment_names: List of attachment filenames.
    """

    raw_bytes: bytes
    headers: dict[str, str | None]
    from_addr: str
    reply_to: str | None
    return_path: str | None
    subject: str
    plain_body: str
    html_body: str
    attachment_names: list[str] = field(default_factory=list)


def ingest_eml_file(path: Path) -> ParsedEmail:
    """Parse a .eml file from disk.

    Args:
        path: Path to the .eml file.

    Returns:
        ParsedEmail instance.
    """
    raw = path.read_bytes()
    return ingest_raw(raw)


def ingest_raw(raw: bytes | str) -> ParsedEmail:
    """Parse raw email bytes or string into a ParsedEmail.

    Args:
        raw: Raw email content (bytes or string).

    Returns:
        ParsedEmail instance.
    """
    if isinstance(raw, str):
        raw_bytes = raw.encode("utf-8", errors="replace")
        msg = email.message_from_string(raw, policy=policy.default)
    else:
        raw_bytes = raw
        msg = email.message_from_bytes(raw, policy=policy.default)

    plain_body, html_body, attachment_names = _extract_parts(msg)
    if not plain_body and html_body:
        plain_body = _html_to_text(html_body)

    parsed = ParsedEmail(
        raw_bytes=raw_bytes,
        headers={k.lower(): _coerce_header_value(v) for k, v in msg.items()},
        from_addr=_coerce_header_value(msg.get("From", "")) or "",
        reply_to=_coerce_header_value(msg.get("Reply-To")),
        return_path=_coerce_header_value(msg.get("Return-Path")),
        subject=_coerce_header_value(msg.get("Subject", "")) or "",
        plain_body=plain_body,
        html_body=html_body,
        attachment_names=attachment_names,
    )

    logger.debug("Ingested email from %s (plain=%d chars, html=%d chars)", parsed.from_addr, len(plain_body), len(html_body))
    return parsed


def _coerce_header_value(value: object | None) -> str | None:
    """Normalize parsed header values into plain strings."""
    if value is None:
        return None
    return str(value)


def _extract_parts(msg: Message) -> tuple[str, str, list[str]]:
    """Recursively extract text, HTML, and attachment parts from a MIME message.

    Args:
        msg: Parsed email Message.

    Returns:
        Tuple of (plain_body, html_body, attachment_names).
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in disposition:
                filename = part.get_filename() or "unnamed"
                attachments.append(filename)
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    plain_parts.append(payload.decode("utf-8", errors="replace"))
            elif content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode("utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    return "\n".join(plain_parts), "\n".join(html_parts), attachments


def _html_to_text(html_body: str) -> str:
    """Best-effort HTML-to-text fallback for HTML-only emails."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.debug("beautifulsoup4 unavailable; using stdlib HTML parser fallback")
        parser = _HTMLTextExtractor()
        parser.feed(html_body)
        text = parser.get_text()
    else:
        text = BeautifulSoup(html_body, "html.parser").get_text(separator=" ", strip=True)

    return re.sub(r"\s+([.,!?;:])", r"\1", text)


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor for environments without BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self._chunks.append(cleaned)

    def get_text(self) -> str:
        return " ".join(self._chunks)
