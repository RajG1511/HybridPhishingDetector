"""Layer 2 URL extraction utilities.

Extracts absolute HTTP(S) URLs from email bodies. The extractor supports both
plain-text URLs and HTML attributes such as ``href``/``src`` while filtering
out unsupported schemes like ``mailto:`` and ``data:``.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(
    r"(?P<url>https?://[^\s\"'<>]+)",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION = ",.;:!?\"'"
_HTML_HINTS = ("<html", "<body", "<a ", "<div", "<table", "<img", "<form")
_SUPPORTED_SCHEMES = {"http", "https"}


def _strip_trailing_punctuation(candidate: str) -> str:
    """Remove punctuation commonly attached to copied URLs.

    Args:
        candidate: Raw regex match or attribute value.

    Returns:
        Cleaned URL candidate.
    """
    cleaned = candidate.strip()

    bracket_pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned

        while cleaned and cleaned[-1] in _TRAILING_PUNCTUATION:
            cleaned = cleaned[:-1]

        for opening, closing in bracket_pairs:
            while cleaned.endswith(closing) and cleaned.count(closing) > cleaned.count(opening):
                cleaned = cleaned[:-1]

    return cleaned


def _is_supported_url(candidate: str) -> bool:
    """Return whether a candidate is an absolute HTTP(S) URL.

    Args:
        candidate: Candidate URL string.

    Returns:
        ``True`` when the candidate is an absolute HTTP(S) URL.
    """
    if not candidate:
        return False

    try:
        parsed = urlparse(candidate)
    except ValueError:
        logger.debug("Skipping malformed URL candidate: %r", candidate)
        return False
    if parsed.scheme.lower() not in _SUPPORTED_SCHEMES:
        return False
    if not parsed.netloc:
        return False
    return True


def _normalize_candidates(candidates: Iterable[str]) -> list[str]:
    """Clean, filter, and deduplicate URL candidates.

    Args:
        candidates: Raw candidate URL strings.

    Returns:
        Ordered list of unique absolute HTTP(S) URLs.
    """
    normalized: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        cleaned = _strip_trailing_punctuation(candidate)
        if not _is_supported_url(cleaned):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)

    return normalized


def extract_urls_from_text(text: str) -> list[str]:
    """Extract absolute HTTP(S) URLs from plain text.

    Args:
        text: Raw plain-text email body.

    Returns:
        Ordered list of unique URLs found in the text.
    """
    if not text:
        return []

    matches = [match.group("url") for match in _URL_PATTERN.finditer(text)]
    urls = _normalize_candidates(matches)
    logger.debug("Extracted %d URLs from plain text", len(urls))
    return urls


def extract_urls_from_html(html: str) -> list[str]:
    """Extract absolute HTTP(S) URLs from HTML.

    The extractor looks at common link-bearing attributes and also scans text
    nodes for pasted plain-text URLs embedded inside HTML content.

    Args:
        html: Raw HTML email body.

    Returns:
        Ordered list of unique URLs found in the HTML.
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for tag in soup.find_all(True):
        for attribute in ("href", "src", "action"):
            value = tag.get(attribute)
            if isinstance(value, str):
                candidates.append(value)

    candidates.extend(match.group("url") for match in _URL_PATTERN.finditer(soup.get_text(" ")))
    urls = _normalize_candidates(candidates)
    logger.debug("Extracted %d URLs from HTML", len(urls))
    return urls


def extract_urls(body: str, is_html: bool | None = None) -> list[str]:
    """Extract URLs from an email body.

    Args:
        body: Email body string.
        is_html: Whether the body should be treated as HTML. When ``None``,
            the function uses a lightweight HTML heuristic.

    Returns:
        Ordered list of unique absolute HTTP(S) URLs.
    """
    if not body:
        return []

    html_mode = is_html
    if html_mode is None:
        lowered = body.lower()
        html_mode = any(marker in lowered for marker in _HTML_HINTS)

    if html_mode:
        return extract_urls_from_html(body)
    return extract_urls_from_text(body)


__all__ = [
    "extract_urls",
    "extract_urls_from_html",
    "extract_urls_from_text",
]
