"""
API Route — Email Analysis

POST /analyze — Accept raw email text or .eml file upload and return a
phishing risk score with XAI explanation.
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from api.schemas import AnalyzeResponse, AnalyzeTextRequest, FeatureAttribution
from src.pipeline.email_ingester import ingest_raw
from src.pipeline.cascade_pipeline import CascadePipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level pipeline instance — initialized at app startup via lifespan
_pipeline: CascadePipeline | None = None


def set_pipeline(pipeline: CascadePipeline) -> None:
    """Register the pipeline instance for use by route handlers.

    Args:
        pipeline: Fully initialized CascadePipeline.
    """
    global _pipeline
    _pipeline = pipeline


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeTextRequest) -> AnalyzeResponse:
    """Analyze a raw email string for phishing.

    Args:
        request: AnalyzeTextRequest with email_text field.

    Returns:
        AnalyzeResponse with risk score, verdict, and XAI explanation.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        parsed = ingest_raw(request.email_text)
        result = _pipeline.run(parsed)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(
        risk_score=result.risk_score,
        verdict=result.verdict,
        narrative=result.narrative,
        layer1_flags=result.layer1_flags,
        layer2_flags=result.layer2_flags,
        layer3_proba=result.layer3_proba,
        layer4_used=result.layer4_used,
        shap_features=[FeatureAttribution(feature=f, weight=w) for f, w in result.shap_features],
        lime_words=[FeatureAttribution(feature=f, weight=w) for f, w in result.lime_words],
    )


@router.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Analyze an uploaded .eml file for phishing.

    Args:
        file: Uploaded .eml file.

    Returns:
        AnalyzeResponse with risk score, verdict, and XAI explanation.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    raw_bytes = await file.read()
    if len(raw_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    try:
        parsed = ingest_raw(raw_bytes)
        result = _pipeline.run(parsed)
    except Exception as exc:
        logger.exception("Analysis failed for uploaded file")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(
        risk_score=result.risk_score,
        verdict=result.verdict,
        narrative=result.narrative,
        layer1_flags=result.layer1_flags,
        layer2_flags=result.layer2_flags,
        layer3_proba=result.layer3_proba,
        layer4_used=result.layer4_used,
        shap_features=[FeatureAttribution(feature=f, weight=w) for f, w in result.shap_features],
        lime_words=[FeatureAttribution(feature=f, weight=w) for f, w in result.lime_words],
    )
