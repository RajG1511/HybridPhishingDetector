"""
Layer 2 — Domain Intelligence

Enriches URL analysis with external domain reputation data:
  - WHOIS domain age lookup (newly registered domains are high-risk)
  - Subdomain / registrar reputation signals

Dependencies: python-whois, dnspython
"""

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def get_domain_age_days(domain: str) -> int | None:
    """Query WHOIS to determine how many days old a domain is.

    Args:
        domain: Fully qualified domain name (e.g. "evil-bank.com").

    Returns:
        Number of days since domain registration, or None if lookup fails.
    """
    try:
        import whois  # type: ignore[import]

        w = whois.whois(domain)
        creation_date = w.creation_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date is None:
            return None

        if creation_date.tzinfo is None:
            creation_date = creation_date.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - creation_date
        return age.days
    except ImportError:
        logger.error("python-whois is not installed; domain age lookup skipped")
        return None
    except Exception as exc:
        logger.warning("WHOIS lookup failed for %s: %s", domain, exc)
        return None


def extract_domain_from_url(url: str) -> str:
    """Extract the registered domain from a URL.

    Args:
        url: Full URL string.

    Returns:
        Domain string (e.g. "example.com"), without subdomains or port.
    """
    netloc = urlparse(url).netloc.lower()
    domain = netloc.split(":")[0]
    # Return last two labels as the registered domain
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def get_domain_features(url: str) -> dict[str, int | None]:
    """Aggregate domain intelligence features for a URL.

    Args:
        url: Full URL string.

    Returns:
        Dictionary with:
            - domain_age_days (int | None): Days since registration.
    """
    domain = extract_domain_from_url(url)
    age = get_domain_age_days(domain)
    return {"domain_age_days": age}
