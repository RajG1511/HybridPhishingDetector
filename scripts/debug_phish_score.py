import sys
import os
import json
import joblib
from pathlib import Path

# Add project root to path
sys.path.append("/home/nikola/projects/HybridPhishingDetector")

from src.pipeline.email_ingester import ingest_raw
from src.pipeline.cascade_pipeline import CascadePipeline
from src.layer3_semantic.ensemble import load_ensemble
from src.layer3_semantic.vectorizer import load_tfidf_vectorizer

def debug_eml(eml_path: str):
    print(f"--- Debugging: {eml_path} ---")
    
    with open(eml_path, "rb") as f:
        raw_bytes = f.read()
    
    parsed = ingest_raw(raw_bytes)
    
    # Load required artifacts for Layer 3 (Semantic)
    print("\nLoading models...")
    try:
        vectorizer = load_tfidf_vectorizer()
        layer3_model = load_ensemble("models/ml/ensemble")
        print("Models loaded successfully.")
    except Exception as exc:
        print(f"Failed to load models: {exc}")
        return

    pipeline = CascadePipeline(
        layer3=layer3_model,
        vectorizer=vectorizer
    )
    result = pipeline.run(parsed)
    
    print(f"\n[FINAL VERDICT]: {result.verdict}")
    print(f"[ENSEMBLE SCORE]: {result.risk_score}%")
    print(f"[METADATA/URL SCORE]: {result.metadata_url_score}%")
    print(f"[SEMANTIC SCORE]: {result.semantic_score}%")
    
    # Check Layer 2 flags
    layer2 = result.layer_outputs.get("layer2", {})
    print(f"\n[LAYER 2 FLAGS]: {layer2.get('url_flags', [])}")
    print(f"[URL COUNT]: {layer2.get('url_count', 0)}")
    
    # Check Layer 1 flags
    layer1 = result.layer_outputs.get("layer1", {})
    print(f"[DKIM STATUS]: {layer1.get('dkim', 'unknown')}")
    print(f"[SPF STATUS]: {layer1.get('spf', 'unknown')}")

if __name__ == "__main__":
    eml_file = "/home/nikola/projects/HybridPhishingDetector/api/temp_email.eml"
    debug_eml(eml_file)
