"""Layer 2 domain intelligence helpers.

This module enriches URL analysis with lightweight WHOIS-based domain age
signals and a persistent JSON cache to avoid repeated network lookups.
"""

from __future__ import annotations

import json
import ipaddress
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

from config.settings import DOMAIN_INTEL_CACHE_PATH, NEW_DOMAIN_THRESHOLD_DAYS

logger = logging.getLogger(__name__)


class DomainIntelFeatures(TypedDict):
    """WHOIS-derived domain metadata for a URL."""

    domain: str
    domain_age_days: int | None
    is_new_domain: bool | None
    lookup_status: str


class _CacheEntry(TypedDict):
    """Persistent cache record for a single domain."""

    domain_age_days: int | None
    checked_at: str
    lookup_status: str


@dataclass(slots=True)
class DomainIntelCache:
    """Simple JSON-backed cache for domain age lookups."""

    cache_path: Path = DOMAIN_INTEL_CACHE_PATH
    _entries: dict[str, _CacheEntry] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._load()

    def get(self, domain: str) -> _CacheEntry | None:
        """Return a cached entry for a domain if available."""
        return self._entries.get(domain.lower())

    def set(
        self,
        domain: str,
        domain_age_days: int | None,
        lookup_status: str,
    ) -> None:
        """Store and persist a cache entry for a domain."""
        normalized = domain.lower()
        self._entries[normalized] = _CacheEntry(
            domain_age_days=domain_age_days,
            checked_at=datetime.now(timezone.utc).isoformat(),
            lookup_status=lookup_status,
        )
        self._persist()

    def _load(self) -> None:
        """Load the cache file if it exists."""
        if not self.cache_path.exists():
            return

        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load domain intel cache %s: %s", self.cache_path, exc)
            return

        if not isinstance(payload, dict):
            logger.warning("Ignoring malformed domain intel cache at %s", self.cache_path)
            return

        for domain, entry in payload.items():
            if not isinstance(domain, str) or not isinstance(entry, dict):
                continue
            self._entries[domain.lower()] = _CacheEntry(
                domain_age_days=entry.get("domain_age_days"),
                checked_at=str(entry.get("checked_at", "")),
                lookup_status=str(entry.get("lookup_status", "cached")),
            )

    def _persist(self) -> None:
        """Persist the cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(self._entries, indent=2, sort_keys=True)
            self.cache_path.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to persist domain intel cache %s: %s", self.cache_path, exc)


def _coerce_creation_date(raw_value: Any) -> datetime | None:
    """Normalize a WHOIS creation-date payload into an aware datetime."""
    value = raw_value[0] if isinstance(raw_value, list) and raw_value else raw_value

    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        candidate = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def extract_domain_from_url(url: str) -> str:
    """Extract a normalized registered domain from a URL.

    Args:
        url: Full absolute URL string.

    Returns:
        Registered domain heuristic (last two labels) or raw hostname/IP.
    """
    hostname = (urlparse(url).hostname or "").lower()
    if not hostname:
        return ""

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return hostname
    except ValueError:
        pass

    labels = [label for label in hostname.split(".") if label]
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return hostname


def get_domain_age_days(
    domain: str,
    *,
    cache: DomainIntelCache | None = None,
    now: datetime | None = None,
) -> int | None:
    """Return the age of a domain in days using WHOIS data.

    Args:
        domain: Registered domain or IP/hostname string.
        cache: Optional cache instance. A default JSON cache is used otherwise.
        now: Optional current timestamp override for testing.

    Returns:
        Domain age in days or ``None`` when unavailable.
    """
    normalized = domain.lower().strip()
    if not normalized:
        return None

    cache_store = cache or DomainIntelCache()
    cached = cache_store.get(normalized)
    if cached is not None:
        return cached["domain_age_days"]

    current_time = now or datetime.now(timezone.utc)

    try:
        import whois  # type: ignore[import]
    except ImportError:
        logger.warning("python-whois is not installed; domain age lookup skipped")
        cache_store.set(normalized, None, "missing_dependency")
        return None

    try:
        result = whois.whois(normalized)
        creation_date = _coerce_creation_date(getattr(result, "creation_date", None))
        if creation_date is None and isinstance(result, dict):
            creation_date = _coerce_creation_date(result.get("creation_date"))

        if creation_date is None:
            cache_store.set(normalized, None, "unknown")
            return None

        age_days = max((current_time - creation_date).days, 0)
        cache_store.set(normalized, age_days, "fetched")
        return age_days
    except Exception as exc:
        logger.warning("WHOIS lookup failed for %s: %s", normalized, exc)
        cache_store.set(normalized, None, "lookup_failed")
        return None


def get_domain_features(
    url: str,
    *,
    cache: DomainIntelCache | None = None,
    now: datetime | None = None,
) -> DomainIntelFeatures:
    """Return domain-age features for a URL.

    Args:
        url: Full absolute URL string.
        cache: Optional cache instance.
        now: Optional current timestamp override for testing.

    Returns:
        Typed dictionary with the registered domain, age, and freshness flag.
    """
    cache_store = cache or DomainIntelCache()
    domain = extract_domain_from_url(url)
    age_days = get_domain_age_days(domain, cache=cache_store, now=now)
    cached_entry = cache_store.get(domain) if domain else None
    lookup_status = cached_entry["lookup_status"] if cached_entry is not None else "unknown"

    return DomainIntelFeatures(
        domain=domain,
        domain_age_days=age_days,
        is_new_domain=None if age_days is None else age_days < NEW_DOMAIN_THRESHOLD_DAYS,
        lookup_status=lookup_status,
    )


__all__ = [
    "DomainIntelCache",
    "DomainIntelFeatures",
    "extract_domain_from_url",
    "get_domain_age_days",
    "get_domain_features",
]
