"""Layer 2 lexical URL feature extraction.

This module computes per-URL lexical features as well as an aggregated
email-level summary across all URLs found in a message.
"""

from __future__ import annotations

import ipaddress
import logging
import math
from collections import Counter
from statistics import fmean
from typing import Final, TypedDict
from urllib.parse import parse_qsl, urlparse

logger = logging.getLogger(__name__)

_SUSPICIOUS_TLDS: Final[set[str]] = {
    "buzz",
    "cf",
    "click",
    "ga",
    "ml",
    "tk",
    "top",
    "xyz",
}
_URL_SHORTENERS: Final[set[str]] = {
    "bit.ly",
    "buff.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "t.co",
    "tiny.one",
    "tinyurl.com",
}
_HOMOGLYPHS: Final[set[str]] = {
    "\u0430",  # Cyrillic small a
    "\u0435",  # Cyrillic small ie
    "\u043E",  # Cyrillic small o
    "\u0440",  # Cyrillic small er
    "\u0441",  # Cyrillic small es
    "\u0443",  # Cyrillic small u
    "\u0456",  # Cyrillic small i
    "\u0458",  # Cyrillic small je
    "\u04CF",  # Cyrillic small palochka
}
_SPECIAL_CHARACTERS: Final[tuple[str, ...]] = ("@", "!", "-", "_")
_CANONICAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "url_length",
    "url_entropy",
    "has_ip",
    "subdomain_depth",
    "path_depth",
    "num_special_chars",
    "suspicious_tld",
    "has_homoglyph",
    "num_url_params",
    "uses_shortener",
    "url_digit_ratio",
    "has_at_symbol",
)


class URLLexicalFeatures(TypedDict):
    """Per-URL lexical features used by Layer 2."""

    url_length: int
    url_entropy: float
    has_ip: bool
    subdomain_depth: int
    path_depth: int
    num_special_chars: int
    suspicious_tld: bool
    has_homoglyph: bool
    num_url_params: int
    uses_shortener: bool
    url_digit_ratio: float
    has_at_symbol: bool
    domain_length: int
    path_length: int
    num_subdomains: int
    has_ip_address: bool
    has_homoglyphs: bool
    has_https: bool


class AggregatedURLLexicalFeatures(TypedDict):
    """Aggregated lexical feature summary for all URLs in an email."""

    url_count: int
    url_length_mean: float
    url_length_max: float
    url_entropy_mean: float
    url_entropy_max: float
    has_ip_mean: float
    has_ip_max: float
    subdomain_depth_mean: float
    subdomain_depth_max: float
    path_depth_mean: float
    path_depth_max: float
    num_special_chars_mean: float
    num_special_chars_max: float
    suspicious_tld_mean: float
    suspicious_tld_max: float
    has_homoglyph_mean: float
    has_homoglyph_max: float
    num_url_params_mean: float
    num_url_params_max: float
    uses_shortener_mean: float
    uses_shortener_max: float
    url_digit_ratio_mean: float
    url_digit_ratio_max: float
    has_at_symbol_mean: float
    has_at_symbol_max: float


def _shannon_entropy(value: str) -> float:
    """Compute the Shannon entropy of a string.

    Args:
        value: Input string.

    Returns:
        Shannon entropy value.
    """
    if not value:
        return 0.0

    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _extract_hostname(url: str) -> str:
    """Return the normalized hostname from a URL.

    Args:
        url: URL string.

    Returns:
        Lowercased hostname without credentials or port.
    """
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def _has_ip_address(hostname: str) -> bool:
    """Return whether a hostname is a raw IP address."""
    if not hostname:
        return False

    candidate = hostname.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def _registered_domain(hostname: str) -> str:
    """Return a simple registered-domain heuristic.

    Args:
        hostname: Lowercased hostname.

    Returns:
        Last two hostname labels when possible, otherwise the hostname itself.
    """
    labels = [label for label in hostname.split(".") if label]
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return hostname


def _subdomain_depth(hostname: str) -> int:
    """Count hostname labels excluding the top-level domain.

    Example:
        ``login.secure.bank.com`` -> ``3``

    Args:
        hostname: Lowercased hostname.

    Returns:
        Host depth excluding the final top-level domain label.
    """
    if not hostname or _has_ip_address(hostname):
        return 0

    labels = [label for label in hostname.split(".") if label]
    return max(len(labels) - 1, 0)


def _traditional_subdomain_count(hostname: str) -> int:
    """Return the number of labels before the registered domain."""
    if not hostname or _has_ip_address(hostname):
        return 0

    labels = [label for label in hostname.split(".") if label]
    return max(len(labels) - 2, 0)


def _path_depth(url: str) -> int:
    """Return the number of non-empty path segments in a URL."""
    path = urlparse(url).path
    return len([segment for segment in path.split("/") if segment])


def _has_homoglyph(url: str, hostname: str) -> bool:
    """Return whether a URL contains common homoglyph indicators."""
    return "xn--" in hostname or any(character in _HOMOGLYPHS for character in url)


def _digit_ratio(url: str) -> float:
    """Return the fraction of characters in a URL that are digits."""
    if not url:
        return 0.0
    return sum(character.isdigit() for character in url) / len(url)


def _empty_aggregate() -> AggregatedURLLexicalFeatures:
    """Return a zero-filled lexical feature summary."""
    return AggregatedURLLexicalFeatures(
        url_count=0,
        url_length_mean=0.0,
        url_length_max=0.0,
        url_entropy_mean=0.0,
        url_entropy_max=0.0,
        has_ip_mean=0.0,
        has_ip_max=0.0,
        subdomain_depth_mean=0.0,
        subdomain_depth_max=0.0,
        path_depth_mean=0.0,
        path_depth_max=0.0,
        num_special_chars_mean=0.0,
        num_special_chars_max=0.0,
        suspicious_tld_mean=0.0,
        suspicious_tld_max=0.0,
        has_homoglyph_mean=0.0,
        has_homoglyph_max=0.0,
        num_url_params_mean=0.0,
        num_url_params_max=0.0,
        uses_shortener_mean=0.0,
        uses_shortener_max=0.0,
        url_digit_ratio_mean=0.0,
        url_digit_ratio_max=0.0,
        has_at_symbol_mean=0.0,
        has_at_symbol_max=0.0,
    )


def extract_lexical_features(url: str) -> URLLexicalFeatures:
    """Extract lexical features from a single URL.

    Args:
        url: Absolute URL string to analyze.

    Returns:
        Typed dictionary with lexical phishing indicators.
    """
    parsed = urlparse(url)
    hostname = _extract_hostname(url)
    registered_domain = _registered_domain(hostname)
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""
    path = parsed.path or ""

    has_ip = _has_ip_address(hostname)
    subdomain_depth = _subdomain_depth(hostname)
    path_depth = _path_depth(url)
    num_special_chars = sum(url.count(character) for character in _SPECIAL_CHARACTERS)
    num_url_params = len(parse_qsl(parsed.query, keep_blank_values=True))
    has_homoglyph = _has_homoglyph(url, hostname)

    features = URLLexicalFeatures(
        url_length=len(url),
        url_entropy=_shannon_entropy(url),
        has_ip=has_ip,
        subdomain_depth=subdomain_depth,
        path_depth=path_depth,
        num_special_chars=num_special_chars,
        suspicious_tld=tld in _SUSPICIOUS_TLDS,
        has_homoglyph=has_homoglyph,
        num_url_params=num_url_params,
        uses_shortener=registered_domain in _URL_SHORTENERS,
        url_digit_ratio=_digit_ratio(url),
        has_at_symbol="@" in url,
        domain_length=len(hostname),
        path_length=len(path),
        num_subdomains=_traditional_subdomain_count(hostname),
        has_ip_address=has_ip,
        has_homoglyphs=has_homoglyph,
        has_https=parsed.scheme.lower() == "https",
    )

    logger.debug("Extracted lexical features for URL: %s", url[:120])
    return features


def aggregate_lexical_features(urls: list[str]) -> AggregatedURLLexicalFeatures:
    """Aggregate lexical features across all URLs found in an email.

    For boolean features, the ``mean`` value represents the proportion of URLs
    where the flag is present and the ``max`` value acts as an ``any`` flag.

    Args:
        urls: Absolute URLs extracted from an email.

    Returns:
        Zero-filled summary when no URLs are present, otherwise a mean/max
        aggregation for each canonical lexical feature.
    """
    if not urls:
        return _empty_aggregate()

    per_url_features = [extract_lexical_features(url) for url in urls]
    summary = _empty_aggregate()
    summary["url_count"] = len(urls)

    for feature_name in _CANONICAL_FEATURE_NAMES:
        numeric_values = [float(features[feature_name]) for features in per_url_features]
        summary[f"{feature_name}_mean"] = fmean(numeric_values)
        summary[f"{feature_name}_max"] = max(numeric_values)

    logger.debug("Aggregated lexical features across %d URLs", len(urls))
    return summary


__all__ = [
    "AggregatedURLLexicalFeatures",
    "URLLexicalFeatures",
    "aggregate_lexical_features",
    "extract_lexical_features",
]
