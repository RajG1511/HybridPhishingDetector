"""
Unit Tests — Layer 1: Protocol Authentication

Tests for header parsing, mismatch detection, DKIM presence checking,
and SPF result parsing.
"""

import pytest
from pathlib import Path

from src.layer1_protocol import analyze_protocol_authentication
from src.layer1_protocol.header_parser import (
    extract_address_domain,
    detect_header_mismatches,
    extract_header_fields,
    parse_eml,
)
from src.layer1_protocol.dkim_verifier import has_dkim_header
from src.layer1_protocol.spf_checker import (
    SPFResult,
    check_spf,
    extract_sender_ip,
    parse_authentication_results_spf,
    parse_received_spf_header,
)


class TestHeaderParser:
    def test_parse_eml_from_path_fixture(self, phishing_eml_path: Path) -> None:
        msg = parse_eml(phishing_eml_path)
        fields = extract_header_fields(msg)
        assert fields["from"] == "Security Team <attacker@evil-bank.com>"
        assert fields["reply_to"] == "collect@phish.net"

    def test_extract_fields_from_phishing_email(self, raw_phishing_email: bytes) -> None:
        msg = parse_eml(raw_phishing_email)
        fields = extract_header_fields(msg)
        assert "attacker@evil-bank.com" in (fields["from"] or "")
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

    def test_extract_address_domain_handles_display_names(self) -> None:
        assert extract_address_domain('"Bank Support" <alerts@bank.example>') == "bank.example"


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

    def test_parse_spf_from_authentication_results(self) -> None:
        header = "mx.example; spf=softfail smtp.mailfrom=bad.example"
        assert parse_authentication_results_spf(header) == SPFResult.SOFTFAIL

    def test_extract_sender_ip_from_received_headers(self) -> None:
        received_headers = [
            "from mail.bad.example (mail.bad.example [203.0.113.10]) by mx.example with ESMTP"
        ]
        assert extract_sender_ip(received_headers) == "203.0.113.10"

    def test_check_spf_matches_ip4_mechanism(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.layer1_protocol.spf_checker.get_spf_record",
            lambda _domain: "v=spf1 ip4:203.0.113.10 -all",
        )
        assert check_spf("example.com", "203.0.113.10") == SPFResult.PASS

    def test_check_spf_hard_fails_non_matching_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.layer1_protocol.spf_checker.get_spf_record",
            lambda _domain: "v=spf1 ip4:203.0.113.10 -all",
        )
        assert check_spf("example.com", "198.51.100.77") == SPFResult.FAIL


class TestLayer1Contract:
    def test_analyze_protocol_authentication_returns_pipeline_contract(self, raw_phishing_email: bytes) -> None:
        result = analyze_protocol_authentication(raw_phishing_email)

        assert set(
            [
                "spf",
                "dkim",
                "arc",
                "header_mismatch",
                "header_issues",
                "metadata_flags",
                "protocol_risk_score",
                "metadata_features",
            ]
        ).issubset(result.keys())
        assert isinstance(result["protocol_risk_score"], int)
        assert isinstance(result["metadata_features"], dict)

    def test_analyze_protocol_authentication_from_eml_path(self, phishing_eml_path: Path) -> None:
        result = analyze_protocol_authentication(phishing_eml_path.read_bytes())
        assert result["header_mismatch"] is True
        assert result["metadata_features"]["sender_ip"] == "203.0.113.10"
