"""
Layer 1 — SPF Record Validator

Validates the Sender Policy Framework (SPF) record for a given sender domain
against the originating IP address. Uses dnspython for DNS TXT record lookups.

Dependencies: dnspython
"""

import logging
import re
from enum import Enum

import dns.resolver

logger = logging.getLogger(__name__)


class SPFResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SOFTFAIL = "softfail"
    NEUTRAL = "neutral"
    NONE = "none"
    TEMPERROR = "temperror"
    PERMERROR = "permerror"


def get_spf_record(domain: str) -> str | None:
    """Fetch the SPF TXT record for a domain via DNS lookup.

    Args:
        domain: The sender domain to query.

    Returns:
        The SPF record string, or None if not found.
    """
    try:
        answers = dns.resolver.resolve(domain, "TXT")
        for rdata in answers:
            txt = b"".join(rdata.strings).decode("utf-8", errors="replace")
            if txt.startswith("v=spf1"):
                return txt
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.DNSException) as exc:
        logger.warning("SPF DNS lookup failed for %s: %s", domain, exc)
    return None


def parse_received_spf_header(header_value: str) -> SPFResult:
    """Parse the Received-SPF header value into an SPFResult enum.

    Args:
        header_value: Raw Received-SPF header string.

    Returns:
        Corresponding SPFResult enum value.
    """
    if not header_value:
        return SPFResult.NONE

    lower = header_value.lower().strip()
    for result in SPFResult:
        if lower.startswith(result.value):
            return result

    return SPFResult.NONE


def check_spf(domain: str, sender_ip: str) -> SPFResult:
    """Perform a basic SPF authorization check.

    Note: Full SPF evaluation (includes, redirects, macros) requires a
    dedicated library such as pyspf. This implementation provides a lightweight
    best-effort check suitable for header-level triage.

    Args:
        domain: The envelope sender domain.
        sender_ip: The IP address of the sending mail server.

    Returns:
        SPFResult indicating the authorization outcome.
    """
    record = get_spf_record(domain)
    if record is None:
        return SPFResult.NONE

    # Check for explicit -all or ~all directives
    if re.search(r"\s-all", record):
        logger.debug("SPF record for %s contains hard fail (-all)", domain)
    elif re.search(r"\s~all", record):
        logger.debug("SPF record for %s contains soft fail (~all)", domain)

    # Placeholder — full IP matching requires pyspf or equivalent
    logger.info("SPF record found for %s; full IP validation not yet implemented", domain)
    return SPFResult.NEUTRAL
