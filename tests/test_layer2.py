"""Unit tests for Layer 2 URL extraction and lexical features."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pytest

from src.layer2_url.domain_intel import (
    DomainIntelCache,
    extract_domain_from_url,
    get_domain_age_days,
    get_domain_features,
)
from src.layer2_url.lexical_features import (
    aggregate_lexical_features,
    extract_lexical_features,
)
from src.layer2_url.url_extractor import (
    extract_urls,
    extract_urls_from_html,
    extract_urls_from_text,
)


class TestURLExtractor:
    def test_extract_from_plain_text_strips_trailing_punctuation(self) -> None:
        text = (
            "Please review https://example.org/reset?token=abc123, "
            "then visit http://evil.com/phish.)"
        )

        urls = extract_urls_from_text(text)

        assert urls == [
            "https://example.org/reset?token=abc123",
            "http://evil.com/phish",
        ]

    def test_extract_from_html_skips_mailto_data_and_relative_urls(self) -> None:
        html = """
        <html>
          <body>
            <a href="https://safe.example.com/login">Safe</a>
            <img src="https://cdn.example.com/pixel.png" />
            <a href="mailto:help@example.com">Support</a>
            <img src="data:image/png;base64,abc123" />
            <a href="/internal/reset">Relative</a>
            Plain text fallback: http://raw.example.net/track?id=1
          </body>
        </html>
        """

        urls = extract_urls_from_html(html)

        assert urls == [
            "https://safe.example.com/login",
            "https://cdn.example.com/pixel.png",
            "http://raw.example.net/track?id=1",
        ]

    def test_extract_urls_auto_detects_html(self) -> None:
        html = '<div><a href="https://portal.example.com">Portal</a></div>'

        assert extract_urls(html, is_html=None) == ["https://portal.example.com"]

    def test_deduplication_preserves_first_seen_order(self) -> None:
        text = "http://evil.com http://evil.com https://legit.org http://evil.com"

        urls = extract_urls_from_text(text)

        assert urls == ["http://evil.com", "https://legit.org"]

    def test_no_urls_returns_empty_list(self) -> None:
        assert extract_urls_from_text("No links here at all.") == []


class TestLexicalFeatures:
    def test_clean_url_features(self) -> None:
        features = extract_lexical_features(
            "https://www.google.com/search?q=tamu&hl=en"
        )

        assert features["has_ip"] is False
        assert features["uses_shortener"] is False
        assert features["suspicious_tld"] is False
        assert features["path_depth"] == 1
        assert features["num_url_params"] == 2
        assert features["has_https"] is True

    def test_sketchy_url_flags_ip_homoglyph_and_at_symbol(self) -> None:
        suspicious_url = (
            "http://192.168.0.1/p\u0430yload-99/login_now!@?a=1&b=2"
        )

        features = extract_lexical_features(suspicious_url)

        assert features["has_ip"] is True
        assert features["has_homoglyph"] is True
        assert features["has_at_symbol"] is True
        assert features["num_special_chars"] >= 4
        assert features["url_digit_ratio"] > 0

    def test_shortener_and_suspicious_tld_detected(self) -> None:
        shortener_features = extract_lexical_features("https://bit.ly/reset123")
        suspicious_tld_features = extract_lexical_features(
            "https://promo-login.xyz/update"
        )

        assert shortener_features["uses_shortener"] is True
        assert suspicious_tld_features["suspicious_tld"] is True

    def test_punycode_hostname_is_treated_as_homoglyph_signal(self) -> None:
        features = extract_lexical_features(
            "http://xn--bcher-kva.example.com/a1b2c3?q=xyz"
        )

        assert features["has_homoglyph"] is True
        assert features["url_entropy"] > 0

    def test_aggregate_features_summary(self) -> None:
        urls = [
            "https://www.google.com/search?q=tamu",
            "http://192.168.0.1/p\u0430yload-99/login_now!@?a=1&b=2",
            "https://promo-login.xyz/update?session=456",
        ]

        summary = aggregate_lexical_features(urls)

        assert summary["url_count"] == 3
        assert summary["has_ip_max"] == pytest.approx(1.0)
        assert summary["has_homoglyph_max"] == pytest.approx(1.0)
        assert summary["suspicious_tld_max"] == pytest.approx(1.0)
        assert summary["num_url_params_max"] == pytest.approx(2.0)
        assert summary["url_length_mean"] > 0

    def test_empty_aggregate_is_zero_filled(self) -> None:
        summary = aggregate_lexical_features([])

        assert summary["url_count"] == 0
        for feature_name, value in summary.items():
            if feature_name == "url_count":
                continue
            assert value == 0.0


class TestDomainIntel:
    def test_extract_domain_basic(self) -> None:
        assert extract_domain_from_url("http://sub.evil-bank.com/path") == "evil-bank.com"

    def test_extract_domain_with_port(self) -> None:
        assert extract_domain_from_url("http://example.com:8080/page") == "example.com"

    def test_domain_age_lookup_uses_cache(self, tmp_path, monkeypatch) -> None:
        fixed_now = datetime(2026, 4, 6, tzinfo=timezone.utc)
        calls: list[str] = []

        class FakeWhoisRecord:
            creation_date = fixed_now - timedelta(days=10)

        class FakeWhoisModule:
            @staticmethod
            def whois(domain: str) -> FakeWhoisRecord:
                calls.append(domain)
                return FakeWhoisRecord()

        monkeypatch.setitem(sys.modules, "whois", FakeWhoisModule)

        cache = DomainIntelCache(cache_path=tmp_path / "domain_age_cache.json")

        first_age = get_domain_age_days("example.com", cache=cache, now=fixed_now)
        second_age = get_domain_age_days("example.com", cache=cache, now=fixed_now)
        features = get_domain_features(
            "https://example.com/login",
            cache=cache,
            now=fixed_now,
        )

        assert first_age == 10
        assert second_age == 10
        assert calls == ["example.com"]
        assert features["is_new_domain"] is True
        assert features["lookup_status"] in {"fetched", "cached"}
        assert cache.cache_path.exists()
