"""
Data Utility — CSV Processor
Handles ingestion and feature mapping for Kaggle-style CSV datasets (Enron, CEAS, etc.)
Mapping Subject/Body text to our 37-feature lexical/metadata schema.
"""

import pandas as pd
import re
import logging
import sys
from pathlib import Path
from typing import Any, List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layer2_url.url_extractor import extract_urls
from src.layer2_url.lexical_features import aggregate_lexical_features

import random
from config.settings import DEFAULT_AUGMENT_RATIO, DEFAULT_COMPROMISED_RATIO

logger = logging.getLogger(__name__)

def process_kaggle_csv(file_path: Path, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Loads a Kaggle CSV and converts each row into a feature dictionary.
    
    Supported column patterns:
    - Text: 'Email Text', 'Message', 'content', 'Body', 'text'
    - Label: 'Email Type', 'Label', 'label', 'status' (0=Safe/Ham, 1=Phishing/Spam)
    """
    if not file_path.exists():
        logger.error(f"CSV file not found: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []

    # 1. Identify Text Column
    text_col = next((c for c in ['Email Text', 'Message', 'content', 'Body', 'text', 'body'] if c in df.columns), None)
    
    # 2. Identify Label Column
    label_col = next((c for c in ['Email Type', 'Label', 'label', 'status', 'Category'] if c in df.columns), None)

    if not text_col or not label_col:
        logger.warning(f"Skipping {file_path.name}: Missing standard columns (Text found: {text_col}, Label found: {label_col})")
        return []

    logger.info(f"Processing {len(df)} rows from {file_path.name}...")
    
    rng = random.Random(seed)
    features_list = []
    
    for _, row in df.iterrows():
        text = str(row[text_col] or "")
        raw_label = str(row[label_col]).lower()
        
        # Normalize Label
        is_phishing = 0
        if any(kw in raw_label for kw in ['phishing', 'spam', '1', 'bad']):
            is_phishing = 1
            
        # Extract URLs and Lexical Features (Layer 2)
        urls = extract_urls(text)
        url_feats = aggregate_lexical_features(urls)
        
        # --- Authentication Simulation ---
        # Instead of 100% perfect or 100% fail, we use our global settings
        # DEFAULT_AUGMENT_RATIO (e.g., 70% of ham gets perfect auth, 30% gets legacy/missing)
        # DEFAULT_COMPROMISED_RATIO (e.g., 10% of phishing gets perfect auth to simulate compromised accounts)
        
        has_auth = False
        if is_phishing == 0:
            if rng.random() < DEFAULT_AUGMENT_RATIO:
                has_auth = True
        else: # is_phishing == 1
            if rng.random() < DEFAULT_COMPROMISED_RATIO:
                has_auth = True
        
        feat_dict = {
            # Protocol Features (Layer 1 - Simulated based on ratios)
            "header_mismatch_count": 0 if has_auth else 1,
            "spf_status": "pass" if has_auth else "none",
            "dkim_status": "pass" if has_auth else "missing",
            "arc_status": "pass" if has_auth else "missing",
            "protocol_risk_score": 0 if has_auth else 30,
            "metadata_flag_count": 0 if has_auth else 2,
            
            # URL Features (Layer 2 - Real)
            "url_count": len(urls),
            "url_flags": [], # Simplified for CSV
            "semantic_available": False,
            "label": is_phishing,
            "_source": file_path.name
        }
        
        # Merge lexical results (url_entropy_max, subdomain_depth_max, etc.)
        feat_dict.update(url_feats)
        
        features_list.append(feat_dict)
        
    return features_list
