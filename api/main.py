"""
API — FastAPI Application Entry Point

Initializes and mounts the FastAPI application, registers routes, and
performs startup initialization of the detection pipeline.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import analyze, health
from config.logging_config import setup_logging
from config.settings import API_HOST, API_PORT

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — runs on startup and shutdown."""
    logger.info("Starting Hybrid Phishing Detector API")

    # Initialize a default pipeline (no models loaded — stubs return score 0)
    from src.pipeline.cascade_pipeline import CascadePipeline
    from api.routes.analyze import set_pipeline

    pipeline = CascadePipeline()
    set_pipeline(pipeline)
    logger.info("Pipeline initialized (no trained models loaded)")

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="Hybrid Phishing Detector",
    description="Multi-layer phishing email detection with XAI explanations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(analyze.router, tags=["analysis"])
