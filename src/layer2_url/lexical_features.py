"""
Layer 2 — Lexical URL Feature Extractor

Computes lexical and structural features from a URL string without making
any network requests. Features include length, entropy, IP presence,
homoglyph characters, subdomain depth, special character counts, and more.

Dependencies: re, math, urllib
"""

import logging
import math
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Common homoglyph characters used in IDN spoofing attacks
_HOMOGLYPHS = set("аеіоурсԁ")  # Cyrillic lookalikes for Latin

_IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"
)


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def extract_lexical_features(url: str) -> dict[str, float | int | bool]:
    """Extract lexical features from a single URL.

    Args:
        url: The URL string to analyze.

    Returns:
        Dictionary of feature name → numeric or boolean value:
            - url_length (int)
            - domain_length (int)
            - path_length (int)
            - num_subdomains (int)
            - has_ip_address (bool)
            - has_https (bool)
            - url_entropy (float)
            - domain_entropy (float)
            - num_digits (int)
            - num_special_chars (int)
            - has_homoglyphs (bool)
            - num_dots (int)
            - has_at_symbol (bool)
            - has_double_slash (bool)
            - has_dash_in_domain (bool)
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path

    # Strip port from domain if present
    domain_no_port = domain.split(":")[0]
    subdomains = domain_no_port.split(".")

    # Remove TLD + second-level domain; remainder = subdomains
    num_subdomains = max(0, len(subdomains) - 2)

    features: dict[str, float | int | bool] = {
        "url_length": len(url),
        "domain_length": len(domain_no_port),
        "path_length": len(path),
        "num_subdomains": num_subdomains,
        "has_ip_address": bool(_IP_PATTERN.match(domain_no_port)),
        "has_https": parsed.scheme == "https",
        "url_entropy": _shannon_entropy(url),
        "domain_entropy": _shannon_entropy(domain_no_port),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_homoglyphs": any(c in _HOMOGLYPHS for c in url),
        "num_dots": url.count("."),
        "has_at_symbol": "@" in url,
        "has_double_slash": "//" in path,
        "has_dash_in_domain": "-" in domain_no_port,
    }

    logger.debug("Extracted %d lexical features for URL: %s", len(features), url[:60])
    return features
