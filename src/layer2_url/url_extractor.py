"""
Layer 2 — URL Extractor

Extracts all hyperlinks from raw email body text (plain text and HTML) using
regex-based pattern matching. Deduplicates and normalizes URLs for downstream
feature extraction.

Dependencies: re, beautifulsoup4
"""

import logging
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>)\]]+",
    re.IGNORECASE,
)


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from plain text using regex.

    Args:
        text: Raw plain-text email body.

    Returns:
        Deduplicated list of URL strings found in the text.
    """
    urls = _URL_PATTERN.findall(text)
    unique = list(dict.fromkeys(urls))
    logger.debug("Extracted %d URLs from plain text", len(unique))
    return unique


def extract_urls_from_html(html: str) -> list[str]:
    """Extract href and src attribute URLs from an HTML email body.

    Args:
        html: Raw HTML string.

    Returns:
        Deduplicated list of absolute URL strings.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []

    for tag in soup.find_all(True):
        for attr in ("href", "src", "action"):
            value = tag.get(attr, "")
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)

    # Also pick up any inline URLs in text nodes
    urls.extend(extract_urls_from_text(soup.get_text()))

    unique = list(dict.fromkeys(urls))
    logger.debug("Extracted %d URLs from HTML", len(unique))
    return unique


def extract_urls(body: str, is_html: bool = False) -> list[str]:
    """Dispatch URL extraction based on content type.

    Args:
        body: Email body string (plain text or HTML).
        is_html: Whether the body is HTML. Auto-detected if not specified.

    Returns:
        Deduplicated list of URLs.
    """
    if is_html or "<html" in body.lower() or "<a " in body.lower():
        return extract_urls_from_html(body)
    return extract_urls_from_text(body)
