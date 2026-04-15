"""Layer 1 SPF parsing and best-effort validation helpers."""

from __future__ import annotations

import ipaddress
import logging
import re
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)

_AUTH_RESULTS_SPF_PATTERN = re.compile(r"\bspf\s*=\s*([a-z_]+)", re.IGNORECASE)
_RECEIVED_IP_PATTERN = re.compile(
    r"(?:\bfrom\b.*?\[|\bclient-ip=)(\d{1,3}(?:\.\d{1,3}){3})",
    re.IGNORECASE,
)


class SPFResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SOFTFAIL = "softfail"
    NEUTRAL = "neutral"
    NONE = "none"
    TEMPERROR = "temperror"
    PERMERROR = "permerror"
    UNKNOWN = "unknown"


def parse_received_spf_header(header_value: str) -> SPFResult:
    """Parse a Received-SPF header into a normalized SPFResult."""
    if not header_value:
        return SPFResult.NONE

    lower = header_value.lower().strip()
    for result in SPFResult:
        if lower.startswith(result.value):
            return result
    return SPFResult.NONE


def parse_authentication_results_spf(header_value: str) -> SPFResult:
    """Parse the SPF status from an Authentication-Results header."""
    if not header_value:
        return SPFResult.NONE

    match = _AUTH_RESULTS_SPF_PATTERN.search(header_value)
    if not match:
        return SPFResult.NONE

    value = match.group(1).strip().lower()
    if value == "hardfail":
        return SPFResult.FAIL
    if value == "unknown":
        return SPFResult.UNKNOWN

    for result in SPFResult:
        if value == result.value:
            return result
    return SPFResult.UNKNOWN


def resolve_spf_status(
    *,
    domain: str,
    sender_ip: str | None = None,
    received_spf_header: str | None = None,
    authentication_results_header: str | None = None,
) -> SPFResult:
    """Resolve SPF status from headers first, then best-effort DNS validation."""
    received_result = parse_received_spf_header(received_spf_header or "")
    if received_result != SPFResult.NONE:
        return received_result

    auth_result = parse_authentication_results_spf(authentication_results_header or "")
    if auth_result != SPFResult.NONE:
        return auth_result

    if domain and sender_ip:
        return check_spf(domain=domain, sender_ip=sender_ip)

    if domain and get_spf_record(domain):
        return SPFResult.NEUTRAL

    return SPFResult.NONE


def extract_sender_ip(received_headers: list[str] | None) -> str | None:
    """Extract the first IPv4 sender IP visible in Received headers."""
    for header in received_headers or []:
        match = _RECEIVED_IP_PATTERN.search(header)
        if match:
            return match.group(1)
    return None


@lru_cache(maxsize=2048)
def get_spf_record(domain: str) -> str | None:
    """Fetch a domain's SPF TXT record."""
    if not domain:
        return None

    try:
        import dns.exception
        import dns.resolver
    except ImportError:
        logger.warning("dnspython is not installed; SPF DNS lookup skipped")
        return None

    try:
        answers = dns.resolver.resolve(domain, "TXT")
        for rdata in answers:
            strings = getattr(rdata, "strings", None)
            if strings:
                txt = b"".join(strings).decode("utf-8", errors="replace")
            else:
                txt = str(rdata).strip('"')
            if txt.startswith("v=spf1"):
                return txt
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.DNSException) as exc:
        logger.debug("SPF DNS lookup failed for %s: %s", domain, exc)
    return None


def check_spf(domain: str, sender_ip: str) -> SPFResult:
    """Best-effort SPF evaluation for a sender IP against a domain SPF record."""
    record = get_spf_record(domain)
    if record is None:
        return SPFResult.NONE

    try:
        ip = ipaddress.ip_address(sender_ip)
    except ValueError:
        logger.warning("Invalid sender IP for SPF check: %s", sender_ip)
        return SPFResult.PERMERROR

    return _evaluate_spf_record(domain=domain, sender_ip=ip, record=record, depth=0)


def _evaluate_spf_record(
    *,
    domain: str,
    sender_ip: ipaddress._BaseAddress,
    record: str,
    depth: int,
) -> SPFResult:
    if depth > 10:
        return SPFResult.PERMERROR

    mechanisms = record.split()[1:]
    redirect_domain: str | None = None

    for mechanism in mechanisms:
        qualifier, body = _split_qualifier(mechanism)

        if "=" in body and not body.startswith(("ip4:", "ip6:", "include:", "exists:")):
            key, value = body.split("=", 1)
            if key == "redirect":
                redirect_domain = value
            continue

        if body == "all":
            return _qualifier_to_result(qualifier)
        if body.startswith("ip4:") or body.startswith("ip6:"):
            if _ip_matches_network(sender_ip, body.split(":", 1)[1]):
                return _qualifier_to_result(qualifier)
            continue
        if body == "a" or body.startswith("a:"):
            host = body.split(":", 1)[1] if ":" in body else domain
            if _matches_a_record(sender_ip, host):
                return _qualifier_to_result(qualifier)
            continue
        if body == "mx" or body.startswith("mx:"):
            host = body.split(":", 1)[1] if ":" in body else domain
            if _matches_mx_record(sender_ip, host):
                return _qualifier_to_result(qualifier)
            continue
        if body.startswith("include:"):
            include_domain = body.split(":", 1)[1]
            include_record = get_spf_record(include_domain)
            if include_record is None:
                continue
            include_result = _evaluate_spf_record(
                domain=include_domain,
                sender_ip=sender_ip,
                record=include_record,
                depth=depth + 1,
            )
            if include_result == SPFResult.PASS:
                return _qualifier_to_result(qualifier)
            if include_result in {SPFResult.TEMPERROR, SPFResult.PERMERROR}:
                return include_result

    if redirect_domain:
        redirect_record = get_spf_record(redirect_domain)
        if redirect_record is None:
            return SPFResult.PERMERROR
        return _evaluate_spf_record(
            domain=redirect_domain,
            sender_ip=sender_ip,
            record=redirect_record,
            depth=depth + 1,
        )

    return SPFResult.NEUTRAL


def _split_qualifier(mechanism: str) -> tuple[str, str]:
    qualifier = "+"
    body = mechanism
    if mechanism[:1] in {"+", "-", "~", "?"}:
        qualifier = mechanism[0]
        body = mechanism[1:]
    return qualifier, body


def _qualifier_to_result(qualifier: str) -> SPFResult:
    if qualifier == "-":
        return SPFResult.FAIL
    if qualifier == "~":
        return SPFResult.SOFTFAIL
    if qualifier == "?":
        return SPFResult.NEUTRAL
    return SPFResult.PASS


def _ip_matches_network(sender_ip: ipaddress._BaseAddress, network_text: str) -> bool:
    try:
        network = ipaddress.ip_network(network_text, strict=False)
    except ValueError:
        return False
    return sender_ip in network


def _matches_a_record(sender_ip: ipaddress._BaseAddress, host: str) -> bool:
    return any(address == sender_ip for address in _resolve_ip_addresses(host, "A"))


def _matches_mx_record(sender_ip: ipaddress._BaseAddress, host: str) -> bool:
    try:
        import dns.exception
        import dns.resolver
    except ImportError:
        return False

    try:
        answers = dns.resolver.resolve(host, "MX")
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.DNSException):
        return False

    for answer in answers:
        exchange = str(answer.exchange).rstrip(".")
        if _matches_a_record(sender_ip, exchange):
            return True
    return False


def _resolve_ip_addresses(host: str, record_type: str) -> list[ipaddress._BaseAddress]:
    try:
        import dns.exception
        import dns.resolver
    except ImportError:
        return []

    try:
        answers = dns.resolver.resolve(host, record_type)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.DNSException):
        return []

    addresses: list[ipaddress._BaseAddress] = []
    for answer in answers:
        try:
            addresses.append(ipaddress.ip_address(str(answer)))
        except ValueError:
            continue
    return addresses


__all__ = [
    "SPFResult",
    "check_spf",
    "extract_sender_ip",
    "get_spf_record",
    "parse_authentication_results_spf",
    "parse_received_spf_header",
    "resolve_spf_status",
]
