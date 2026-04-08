"""API request and response schemas."""

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    """Request body for analyzing a raw email string."""

    email_text: str = Field(..., min_length=1, description="Raw email string (RFC 2822)")


class FeatureAttribution(BaseModel):
    """Reserved feature attribution entry for future XAI expansion."""

    feature: str
    weight: float


class AnalyzeResponse(BaseModel):
    """Response from the analysis endpoints."""

    risk_score: int = Field(..., ge=0, le=100)
    verdict: str = Field(..., description='One of "safe", "suspicious", or "phishing".')
    narrative: str
    layer1_flags: list[str] = []
    layer2_flags: list[str] = []
    layer3_proba: float = 0.0
    layer4_used: bool = False
    shap_features: list[FeatureAttribution] = []
    lime_words: list[FeatureAttribution] = []


class HealthResponse(BaseModel):
    """Response from the health endpoint."""

    status: str = "ok"
    version: str = "0.1.0"
