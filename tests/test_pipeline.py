"""
Integration Tests — Full Cascade Pipeline

Tests the end-to-end pipeline from raw email ingestion through risk scoring,
using synthetic emails without requiring trained models.
"""

import pytest

from src.pipeline.email_ingester import ingest_raw
from src.pipeline.cascade_pipeline import CascadePipeline, _score_to_verdict
from src.pipeline.risk_scorer import RiskScorer, RiskSignals


class TestEmailIngester:
    def test_ingest_raw_bytes(self, raw_phishing_email: bytes) -> None:
        parsed = ingest_raw(raw_phishing_email)
        assert parsed.from_addr == "attacker@evil-bank.com"
        assert parsed.subject == "Urgent: Verify your account"
        assert len(parsed.plain_body) > 0

    def test_ingest_raw_string(self, phishing_email_text: str) -> None:
        parsed = ingest_raw(phishing_email_text)
        assert parsed.reply_to == "collect@phish.net"

    def test_ingest_ham_email(self, raw_ham_email: bytes) -> None:
        parsed = ingest_raw(raw_ham_email)
        assert "alice@company.com" in parsed.from_addr


class TestRiskScorer:
    def test_zero_signals_gives_low_score(self) -> None:
        scorer = RiskScorer()
        signals = RiskSignals()
        score = scorer.compute(signals)
        assert score == 0

    def test_all_failures_gives_high_score(self) -> None:
        scorer = RiskScorer()
        signals = RiskSignals(
            header_mismatch_count=3,
            dkim_valid=False,
            spf_pass=False,
            url_flags=["IP-based URL", "Homoglyph URL"],
            ensemble_proba=0.95,
        )
        score = scorer.compute(signals)
        assert score >= 75

    def test_score_capped_at_100(self) -> None:
        scorer = RiskScorer()
        signals = RiskSignals(
            header_mismatch_count=100,
            dkim_valid=False,
            spf_pass=False,
            url_flags=["a"] * 50,
            ensemble_proba=1.0,
        )
        assert scorer.compute(signals) == 100


class TestCascadePipeline:
    def test_pipeline_runs_without_models(self, raw_phishing_email: bytes) -> None:
        pipeline = CascadePipeline()
        parsed = ingest_raw(raw_phishing_email)
        result = pipeline.run(parsed)
        assert 0 <= result.risk_score <= 100
        assert result.verdict in ("benign", "suspicious", "phishing")

    def test_verdict_mapping(self) -> None:
        assert _score_to_verdict(0) == "benign"
        assert _score_to_verdict(39) == "benign"
        assert _score_to_verdict(40) == "suspicious"
        assert _score_to_verdict(74) == "suspicious"
        assert _score_to_verdict(75) == "phishing"
        assert _score_to_verdict(100) == "phishing"
