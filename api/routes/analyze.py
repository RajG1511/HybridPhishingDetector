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
    layer4 = result.layer_outputs.get("layer4", {})

    layer2_flags = list(layer2.get("url_flags", []))
    layer2_url_count = int(layer2.get("url_count", 0))
    layer2_feature_summary = dict(layer2.get("feature_summary", {}))

    layer3_label = str(layer3.get("predicted_label", "unknown"))
    layer3_proba = float(layer3.get("phishing_probability", layer3.get("confidence", 0.0)))

    return AnalyzeResponse(
        risk_score=result.risk_score,
        verdict=result.verdict,
        narrative=result.explanation,

        # Layer 1
        layer1_spf=str(layer1.get("spf", "unknown")),
        layer1_dkim=str(layer1.get("dkim", "unknown")),
        layer1_arc=str(layer1.get("arc", "unknown")),
        layer1_header_mismatch=bool(layer1.get("header_mismatch", False)),
        layer1_protocol_risk_score=int(layer1.get("protocol_risk_score", 0)),
        layer1_header_issues=list(layer1.get("header_issues", [])),
        layer1_metadata_flags=list(layer1.get("metadata_flags", [])),

        # Layer 2
        layer2_flags=layer2_flags,
        layer2_url_count=layer2_url_count,
        layer2_feature_summary=layer2_feature_summary,

        # Layer 3
        layer3_label=layer3_label,
        layer3_proba=layer3_proba,

        # Layer 4
        layer4_used=result.layer4_used,
        layer4_eligible=bool(layer4.get("eligible", False)),
        layer4_reason=str(layer4.get("reason", "")),
        layer4_note=str(layer4.get("note", "")),

        # XAI
        shap_features=getattr(result, "shap_features", []),
        lime_words=getattr(result, "lime_words", []),
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