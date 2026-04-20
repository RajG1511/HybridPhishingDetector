import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.email_ingester import ingest_eml_file
from src.layer1_protocol.metadata_features import extract_metadata_features
from src.layer2_url.url_extractor import extract_urls
from src.layer2_url.lexical_features import aggregate_lexical_features
from src.pipeline.risk_scorer import RiskScorer, RiskSignals
from src.pipeline.metadata_url_model import MetadataURLModel, build_feature_dict_from_signals
from config.settings import METADATA_URL_MODEL_PATH

def debug_email(eml_path: str):
    print(fDebug
