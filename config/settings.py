"""
Global Configuration Settings

Centralizes all project-wide configuration: file paths, model hyperparameters,
ensemble thresholds, and API settings. Import from this module rather than
hardcoding values elsewhere.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# === Model Hyperparameters ===
TFIDF_MAX_FEATURES = 10_000
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
BILSTM_HIDDEN_DIM = 128
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.3
BILSTM_EPOCHS = 20
BILSTM_BATCH_SIZE = 64
BILSTM_LEARNING_RATE = 1e-3

# === Ensemble Thresholds ===
GREY_ZONE_LOW = 0.40    # Below this → benign
GREY_ZONE_HIGH = 0.75   # Above this → phishing
# Between LOW and HIGH → escalate to RAG layer

# === API ===
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_INFERENCE_LATENCY_MS = 200
