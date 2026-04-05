"""
Unit Tests — Layer 1: Protocol Authentication

Tests for header parsing, mismatch detection, DKIM presence checking,
and SPF result parsing.
"""

import pytest

from src.layer1_protocol.header_parser import (
    detect_header_mismatches,
    extract_header_fields,
    parse_eml,
)
from src.layer1_protocol.dkim_verifier import has_dkim_header
from src.layer1_protocol.spf_checker import parse_received_spf_header, SPFResult


class TestHeaderParser:
    def test_extract_fields_from_phishing_email(self, raw_phishing_email: bytes) -> None:
        msg = parse_eml(raw_phishing_email)
        fields = extract_header_fields(msg)
        assert fields["from"] == "attacker@evil-bank.com"
        assert fields["reply_to"] == "collect@phish.net"

    def test_detect_reply_to_mismatch(self, raw_phishing_email: bytes) -> None:
        msg = parse_eml(raw_phishing_email)
        fields = extract_header_fields(msg)
        mismatches = detect_header_mismatches(fields)
        assert any("Reply-To" in m for m in mismatches)

    def test_no_mismatch_on_clean_email(self, raw_ham_email: bytes) -> None:
        msg = parse_eml(raw_ham_email)
        fields = extract_header_fields(msg)
        mismatches = detect_header_mismatches(fields)
        assert len(mismatches) == 0


class TestDKIMChecker:
    def test_no_dkim_header_in_synthetic_email(self, raw_phishing_email: bytes) -> None:
        assert not has_dkim_header(raw_phishing_email)

    def test_dkim_header_detected(self) -> None:
        raw = b"DKIM-Signature: v=1; a=rsa-sha256;\r\nFrom: test@example.com\r\n\r\nBody"
        assert has_dkim_header(raw)


class TestSPFChecker:
    def test_parse_spf_pass(self) -> None:
        assert parse_received_spf_header("pass (domain of test@example.com)") == SPFResult.PASS

    def test_parse_spf_fail(self) -> None:
        assert parse_received_spf_header("fail (not authorized)") == SPFResult.FAIL

    def test_parse_spf_none_on_empty(self) -> None:
        assert parse_received_spf_header("") == SPFResult.NONE
