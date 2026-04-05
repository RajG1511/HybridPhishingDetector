"""
API — Pydantic Request/Response Schemas

Defines the data models for all FastAPI endpoints. Input accepts either
a raw email string or a base64-encoded .eml file. Output includes the
risk score, verdict, and XAI explanation.
"""

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    """Request body for analyzing a raw email text string.

    Attributes:
        email_text: Raw email content (headers + body, RFC 2822 format).
    """

    email_text: str = Field(..., min_length=1, description="Raw email string (RFC 2822)")


class FeatureAttribution(BaseModel):
    """Single feature attribution for XAI output.

    Attributes:
        feature: Feature name or word.
        weight: Signed attribution weight (positive = phishing signal).
    """

    feature: str
    weight: float


class AnalyzeResponse(BaseModel):
    """Response from the /analyze endpoint.

    Attributes:
        risk_score: Integer 0–100 phishing risk score.
        verdict: One of "benign", "suspicious", "phishing".
        narrative: Human-readable XAI explanation.
        layer1_flags: Protocol authentication issues.
        layer2_flags: URL analysis risk flags.
        layer3_proba: Ensemble phishing probability.
        layer4_used: Whether the RAG layer was invoked.
        shap_features: Top SHAP feature attributions.
        lime_words: Top LIME word attributions.
    """

    risk_score: int = Field(..., ge=0, le=100)
    verdict: str
    narrative: str
    layer1_flags: list[str] = []
    layer2_flags: list[str] = []
    layer3_proba: float = 0.0
    layer4_used: bool = False
    shap_features: list[FeatureAttribution] = []
    lime_words: list[FeatureAttribution] = []


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = "ok"
    version: str = "0.1.0"
