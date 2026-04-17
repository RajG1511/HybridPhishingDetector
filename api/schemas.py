"""API request and response schemas."""

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    email_text: str = Field(..., min_length=1, description="Raw email string (RFC 2822)")


class FeatureAttribution(BaseModel):
    feature: str
    weight: float


class AnalyzeResponse(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    verdict: str = Field(..., description='One of "safe", "suspicious", or "phishing".')
    narrative: str

    # Layer 1
    layer1_spf: str = "unknown"
    layer1_dkim: str = "unknown"
    layer1_arc: str = "unknown"
    layer1_header_mismatch: bool = False
    layer1_protocol_risk_score: int = 0
    layer1_header_issues: list[str] = []
    layer1_metadata_flags: list[str] = []

    # Layer 2
    layer2_flags: list[str] = []
    layer2_url_count: int = 0
    layer2_feature_summary: dict = {}

    # Layer 3
    layer3_label: str = "unknown"
    layer3_proba: float = 0.0

    # Layer 4
    layer4_used: bool = False
    layer4_eligible: bool = False
    layer4_reason: str = ""
    layer4_note: str = ""

    # XAI
    shap_features: list[FeatureAttribution] = []
    lime_words: list[FeatureAttribution] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"