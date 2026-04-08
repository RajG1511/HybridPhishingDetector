"""
API Route — Email Analysis

POST /analyze — Accept raw email text or .eml file upload and return a
phishing risk score with XAI explanation.
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from api.schemas import AnalyzeResponse, AnalyzeTextRequest
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


def _build_response(result) -> AnalyzeResponse:
    """Convert a DetectionResult into the current API response schema."""
    layer1 = result.layer_outputs.get("layer1", {})
    layer2 = result.layer_outputs.get("layer2", {})
    layer3 = result.layer_outputs.get("layer3", {})

    layer1_flags = list(layer1.get("header_issues", []))
    if not layer1_flags:
        layer1_flags = list(layer1.get("metadata_flags", []))

    layer2_flags = list(layer2.get("url_flags", []))
    layer3_proba = float(layer3.get("phishing_probability", 0.0))

    return AnalyzeResponse(
        risk_score=result.risk_score,
        verdict=result.verdict,
        narrative=result.explanation,
        layer1_flags=layer1_flags,
        layer2_flags=layer2_flags,
        layer3_proba=layer3_proba,
        layer4_used=result.layer4_used,
        shap_features=[],
        lime_words=[],
    )


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

    return _build_response(result)


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

    return _build_response(result)
