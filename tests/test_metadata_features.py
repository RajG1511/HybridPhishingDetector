"""Unit tests for Layer 1 metadata feature extraction."""

from __future__ import annotations

from src.layer1_protocol.metadata_features import extract_metadata_features


class TestMetadataFeatures:
    def test_phishing_email_metadata_flags_sender_mismatch(self, raw_phishing_email: bytes) -> None:
        features = extract_metadata_features(raw_phishing_email)

        assert features.from_domain == "evil-bank.com"
        assert features.reply_to_domain == "phish.net"
        assert features.from_reply_to_mismatch is True
        assert features.from_return_path_mismatch is False
        assert features.dkim == "missing"
        assert features.spf == "none"
        assert features.protocol_risk_score > 0
        assert "Reply-To domain mismatch" in features.metadata_flags

    def test_ham_email_metadata_is_lower_risk(self, raw_ham_email: bytes, raw_phishing_email: bytes) -> None:
        ham_features = extract_metadata_features(raw_ham_email)
        phishing_features = extract_metadata_features(raw_phishing_email)

        assert ham_features.from_reply_to_mismatch is False
        assert ham_features.from_return_path_mismatch is False
        assert ham_features.protocol_risk_score < phishing_features.protocol_risk_score

    def test_message_id_and_spf_headers_are_parsed(self) -> None:
        raw_email = """\
From: billing@company.com
Reply-To: billing@company.com
Return-Path: <billing@company.com>
Message-ID: <12345@alerts.bad.net>
Received-SPF: pass (domain of company.com designates 1.2.3.4 as permitted sender)
Authentication-Results: mx.example; spf=pass smtp.mailfrom=company.com
Received: from mail.company.com (mail.company.com [1.2.3.4]) by mx.example
Subject: Monthly invoice
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Invoice attached.
"""

        features = extract_metadata_features(raw_email)

        assert features.spf == "pass"
        assert features.has_authentication_results is True
        assert features.num_received_headers == 1
        assert features.sender_ip == "1.2.3.4"
        assert features.message_id_domain == "alerts.bad.net"
        assert features.message_id_domain_mismatch is True
