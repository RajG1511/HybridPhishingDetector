import logging
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI

from api.routes import analyze, health
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hybrid Phishing Detector API")

    from src.pipeline.cascade_pipeline import CascadePipeline

    layer3_model = None
    vectorizer = None
    lime_explainer = None
    shap_explainer = None
    narrative_generator = None

    try:
        from src.layer3_semantic.ensemble import load_ensemble
        from src.layer3_semantic.vectorizer import load_tfidf_vectorizer
        from src.xai.lime_explainer import LIMEExplainer
        from src.xai.shap_explainer import SHAPExplainer
        from src.xai.narrative_generator import generate_explanation

        vectorizer = load_tfidf_vectorizer()
        layer3_model = load_ensemble("models/ml/ensemble")

        lime_explainer = LIMEExplainer(vectorizer, layer3_model)

        rf_model = joblib.load("models/ml/random_forest.joblib")
        shap_explainer = SHAPExplainer(rf_model, vectorizer)

        narrative_generator = generate_explanation

        logger.info("Loaded Layer 3 TF-IDF vectorizer, ensemble, and XAI explainers")
    except Exception as exc:
        logger.warning(
            "Could not load Layer 3/XAI artifacts; semantic layer will run in neutral mode: %s",
            exc,
        )

    pipeline = CascadePipeline(
        layer3=layer3_model,
        vectorizer=vectorizer,
        lime_explainer=lime_explainer,
        shap_explainer=shap_explainer,
        narrative_generator=narrative_generator,
    )
    analyze.set_pipeline(pipeline)

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