"""
Unit Tests — Layer 2: URL Feature Extraction

Tests for URL extraction from plain text and HTML, lexical feature computation,
and domain extraction utilities.
"""

import pytest

from src.layer2_url.url_extractor import extract_urls_from_text, extract_urls_from_html, extract_urls
from src.layer2_url.lexical_features import extract_lexical_features
from src.layer2_url.domain_intel import extract_domain_from_url


class TestURLExtractor:
    def test_extract_from_plain_text(self) -> None:
        text = "Click here: http://evil.com/phish and also https://legit.org/page"
        urls = extract_urls_from_text(text)
        assert "http://evil.com/phish" in urls
        assert "https://legit.org/page" in urls

    def test_extract_from_html(self) -> None:
        html = '<a href="http://phish.net/login">Click</a>'
        urls = extract_urls_from_html(html)
        assert "http://phish.net/login" in urls

    def test_deduplication(self) -> None:
        text = "http://evil.com http://evil.com http://evil.com"
        urls = extract_urls_from_text(text)
        assert len(urls) == 1

    def test_no_urls_returns_empty(self) -> None:
        assert extract_urls_from_text("no links here at all") == []


class TestLexicalFeatures:
    def test_ip_address_url_detected(self) -> None:
        feats = extract_lexical_features("http://192.168.1.1/login")
        assert feats["has_ip_address"] is True

    def test_https_detected(self) -> None:
        feats = extract_lexical_features("https://example.com/page")
        assert feats["has_https"] is True

    def test_subdomain_count(self) -> None:
        feats = extract_lexical_features("http://a.b.evil.com/path")
        assert feats["num_subdomains"] == 2

    def test_at_symbol_flag(self) -> None:
        feats = extract_lexical_features("http://evil.com@legit.com/phish")
        assert feats["has_at_symbol"] is True

    def test_entropy_nonzero_for_complex_url(self) -> None:
        feats = extract_lexical_features("http://xn--bcher-kva.example.com/a1b2c3?q=xyz")
        assert feats["url_entropy"] > 0


class TestDomainIntel:
    def test_extract_domain_basic(self) -> None:
        assert extract_domain_from_url("http://sub.evil-bank.com/path") == "evil-bank.com"

    def test_extract_domain_with_port(self) -> None:
        assert extract_domain_from_url("http://example.com:8080/page") == "example.com"
