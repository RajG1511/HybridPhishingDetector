"""Global configuration settings for the phishing detector project."""

from pathlib import Path
import os

# === Pipeline Augmentation ===
DEFAULT_AUGMENT_RATIO = float(os.getenv("EML_AUGMENT_RATIO", "0.70"))
DEFAULT_COMPROMISED_RATIO = float(os.getenv("EML_COMPROMISED_RATIO", "0.10"))

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"

# Raw .eml dataset directories
EPVME_DATA_DIR = RAW_DIR / "epvme" / "eml"
SPAMASSASSIN_DATA_DIR = RAW_DIR / "spamassassin" / "eml"

# Processed feature tables
EML_TRAINING_FEATURES_PATH = FEATURES_DIR / "eml_training_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_URL_MODELS_DIR = MODELS_DIR / "metadata_url"

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
GREY_ZONE_LOW = 0.30
GREY_ZONE_HIGH = 0.47

# === Layer 2 / Risk Scoring ===
NEW_DOMAIN_THRESHOLD_DAYS = 30
DOMAIN_INTEL_CACHE_PATH = FEATURES_DIR / "domain_age_cache.json"
LAYER1_MAX_RISK_POINTS = 30
LAYER2_MAX_RISK_POINTS = 25
LAYER3_MAX_RISK_POINTS = 45
SAFE_RISK_THRESHOLD = int(GREY_ZONE_LOW * 100)
PHISHING_RISK_THRESHOLD = int(GREY_ZONE_HIGH * 100)
METADATA_URL_MODEL_PATH = METADATA_URL_MODELS_DIR / "metadata_url_model.joblib"
METADATA_URL_MODEL_METRICS_PATH = FEATURES_DIR / "metadata_url_model_metrics.json"
METADATA_URL_SPLIT_MANIFEST_PATH = FEATURES_DIR / "metadata_url_split_manifest.csv"

# === API ===
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_INFERENCE_LATENCY_MS = 200
