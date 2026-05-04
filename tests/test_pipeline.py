"""Integration tests for the cascade pipeline and risk scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import ENSEMBLE_PHISHING_THRESHOLD as PHISHING_RISK_THRESHOLD, ENSEMBLE_SAFE_THRESHOLD as SAFE_RISK_THRESHOLD
from src.pipeline.cascade_pipeline import CascadePipeline, _score_to_verdict
from src.pipeline.email_ingester import ingest_eml_file, ingest_raw
from src.pipeline.risk_scorer import RiskScorer, RiskSignals


class TestEmailIngester:
    def test_ingest_eml_file_from_fixture(self, phishing_eml_path: Path) -> None:
        parsed = ingest_eml_file(phishing_eml_path)
        assert "attacker@evil-bank.com" in parsed.from_addr
        assert parsed.reply_to == "collect@phish.net"
        assert parsed.plain_body

    def test_ingest_raw_bytes(self, raw_phishing_email: bytes) -> None:
        parsed = ingest_raw(raw_phishing_email)
        assert "attacker@evil-bank.com" in parsed.from_addr
        assert parsed.subject == "Urgent: Verify your account"
        assert len(parsed.plain_body) > 0

    def test_ingest_raw_string(self, phishing_email_text: str) -> None:
        parsed = ingest_raw(phishing_email_text)
        assert parsed.reply_to == "collect@phish.net"

    def test_ingest_ham_email(self, raw_ham_email: bytes) -> None:
        parsed = ingest_raw(raw_ham_email)
        assert "alice@company.com" in parsed.from_addr

    def test_ingest_html_only_email_generates_plain_text_fallback(
        self,
        html_only_email_text: str,
    ) -> None:
        parsed = ingest_raw(html_only_email_text)
        assert "Hello team." in parsed.plain_body


class TestRiskScorer:
    def test_zero_signals_gives_low_score(self) -> None:
        scorer = RiskScorer()
        signals = RiskSignals()

        score = scorer.compute(signals)
        result = scorer.score(signals)

        assert score == 0
        assert result.label == "safe"

    def test_all_failures_gives_high_score(self) -> None:
        scorer = RiskScorer()
        signals = RiskSignals(
            header_mismatch_count=3,
            dkim_valid=False,
            spf_pass=False,
            url_flags=["IP-based URL", "Homoglyph URL"],
            url_feature_summary={
                "has_ip_max": 1.0,
                "has_homoglyph_max": 1.0,
                "suspicious_tld_max": 1.0,
            },
            ensemble_proba=0.95,
        )

        result = scorer.score(signals)

        assert result.score >= 75
        assert result.label == "phishing"

    def test_score_capped_at_100(self) -> None:
        scorer = RiskScorer()
        signals = RiskSignals(
            protocol_risk_score=100,
            url_flags=["a"] * 50,
            ensemble_proba=1.0,
        )

        assert scorer.compute(signals) == 100


class TestCascadePipeline:
    def test_pipeline_runs_without_models(self, raw_phishing_email: bytes) -> None:
        pipeline = CascadePipeline(metadata_url_model_path=None)
        parsed = ingest_raw(raw_phishing_email)

        result = pipeline.run(parsed)

        assert 0 <= result.risk_score <= 100
        assert result.verdict in ("safe", "suspicious", "phishing")
        assert result.predicted_label in ("unknown", "legitimate", "phishing_human", "phishing_ai")
        assert "layer1" in result.layer_outputs
        assert "layer2" in result.layer_outputs
        assert "layer3" in result.layer_outputs
        assert "layer4" in result.layer_outputs
        assert "metadata_features" in result.layer_outputs["layer1"]
        assert result.layer_outputs["layer4"]["status"] == "placeholder"
        assert result.layer_outputs["layer4"]["used"] is False
        assert result.layer4_used is False
        assert result.explanation

    def test_pipeline_integrates_mocked_layers(self, raw_phishing_email: bytes) -> None:
        def mock_layer1(_parsed) -> dict[str, object]:
            return {
                "spf": "fail",
                "dkim": "fail",
                "arc": "missing",
                "header_mismatch": True,
                "header_issues": ["Reply-To mismatch"],
                "protocol_risk_score": 65,
            }

        def mock_layer3(_parsed) -> dict[str, object]:
            return {
                "predicted_label": "phishing_ai",
                "confidence": 0.91,
                "class_probabilities": {
                    "legitimate": 0.04,
                    "phishing_human": 0.05,
                    "phishing_ai": 0.91,
                },
            }

        pipeline = CascadePipeline(
            layer1=mock_layer1,
            layer3=mock_layer3,
            metadata_url_model_path=None,
        )

        result = pipeline.run(ingest_raw(raw_phishing_email))

        assert result.predicted_label == "phishing_ai"
        assert result.confidence == pytest.approx(0.96)
        assert result.layer_outputs["layer1"]["protocol_risk_score"] == 65
        assert result.layer_outputs["layer2"]["url_count"] >= 1
        assert result.risk_score >= 60
        assert result.layer_outputs["layer4"]["eligible"] == (
            SAFE_RISK_THRESHOLD <= result.risk_score < PHISHING_RISK_THRESHOLD
        )
        assert result.layer_outputs["layer4"]["status"] == "placeholder"
        assert result.verdict in ("suspicious", "phishing")

    def test_verdict_mapping(self) -> None:
        assert _score_to_verdict(0) == "safe"
        assert _score_to_verdict(SAFE_RISK_THRESHOLD - 1) == "safe"
        assert _score_to_verdict(SAFE_RISK_THRESHOLD) == "suspicious"
        assert _score_to_verdict(PHISHING_RISK_THRESHOLD - 1) == "suspicious"
        assert _score_to_verdict(PHISHING_RISK_THRESHOLD) == "phishing"
        assert _score_to_verdict(100) == "phishing"
