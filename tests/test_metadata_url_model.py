"""Tests for the learned metadata + URL model helpers."""

from __future__ import annotations

from src.pipeline.metadata_url_model import MetadataURLModel, build_feature_dict_from_signals
from src.pipeline.risk_scorer import RiskScorer, RiskSignals


class _DummyClassifier:
    def predict_proba(self, rows):
        return [[0.25, 0.75] for _ in rows]


class TestMetadataURLModel:
    def test_feature_builder_encodes_statuses_and_summary(self) -> None:
        signals = RiskSignals(
            header_mismatch_count=2,
            spf_status="none",
            dkim_status="missing",
            arc_status="fail",
            protocol_risk_score=42,
            metadata_flag_count=5,
            url_count=3,
            url_flags=["IP-based URL", "Homoglyph URL"],
            url_feature_summary={
                "url_entropy_max": 4.5,
                "has_ip_max": 1.0,
                "url_digit_ratio_max": 0.2,
            },
        )

        feature_dict = build_feature_dict_from_signals(signals)

        assert feature_dict["protocol_risk_score"] == 42.0
        assert feature_dict["metadata_flag_count"] == 5.0
        assert feature_dict["has_ip_max"] == 1.0
        assert feature_dict["spf_none"] == 1.0
        assert feature_dict["dkim_missing"] == 1.0
        assert feature_dict["arc_fail"] == 1.0

    def test_risk_scorer_can_use_learned_metadata_url_model(self) -> None:
        learned_model = MetadataURLModel(
            model=_DummyClassifier(),
            threshold=0.5,
            model_name="dummy_metadata_url",
        )
        scorer = RiskScorer(
            metadata_url_model=learned_model,
            metadata_url_model_path=None,
        )
        signals = RiskSignals(
            header_mismatch_count=1,
            spf_status="none",
            dkim_status="missing",
            arc_status="missing",
            protocol_risk_score=60,
            metadata_flag_count=4,
            url_count=1,
            url_flags=["IP-based URL"],
            url_feature_summary={"has_ip_max": 1.0},
            semantic_available=False,
        )

        result = scorer.score(signals)

        assert result.metadata_url_model_name == "dummy_metadata_url"
        assert result.metadata_url_probability == 0.75
        assert result.layer_scores["layer1_protocol"] + result.layer_scores["layer2_url"] == 42
        assert result.layer_scores["layer3_semantic"] == 0
