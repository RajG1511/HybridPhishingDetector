import os
import sys
import argparse
import logging
import json
import joblib
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_all_datasets
from src.layer3_semantic.preprocessor import process_dataframe
from src.layer3_semantic.vectorizer import fit_tfidf, transform_tfidf, save_tfidf_vectorizer, save_tfidf_features, get_distilbert_embeddings
from src.layer3_semantic.ml_models import build_logistic_regression, build_random_forest, build_svm, build_xgboost
from src.layer3_semantic.ensemble import SuperLearner
from src.layer3_semantic.bilstm_model import train_bilstm

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_stage_metadata_training():
    logger.info("=== STAGE: Metadata plus URL Model Training ===")
    try:
        from scripts.train_metadata_url_model import load_dataset, create_split_manifest, build_estimator, MetadataURLModel
        from src.pipeline.metadata_url_model import DEFAULT_FEATURE_NAMES
        from config.settings import METADATA_URL_MODEL_PATH
        
        # We target the existing extracted features for this specific model
        results_csv = PROJECT_ROOT / 'data/processed/features/eml_training_features.csv'
        if not results_csv.exists():
            logger.warning("Skipping Metadata training: 'eml_training_features.csv' not found.")
            return

        dataset = load_dataset(results_csv)
        
        # Train XGBoost as determined in previous session
        estimator = build_estimator('xgboost', random_state=42)
        
        # Simplified train for unified pipeline integration
        X = dataset[list(DEFAULT_FEATURE_NAMES)].fillna(0.0)
        y = dataset['label'].astype(int)
        
        logger.info("Fitting Metadata XGBoost model...")
        estimator.fit(X, y)
        
        bundle = MetadataURLModel(
            model=estimator,
            feature_names=list(DEFAULT_FEATURE_NAMES),
            threshold=0.47, # Validated optimal threshold
            model_name='xgboost_metadata_url',
            training_metrics={'status': 'trained_via_unified_pipeline'}
        )
        bundle.save(METADATA_URL_MODEL_PATH)
        logger.info(f"✓ Metadata model saved to {METADATA_URL_MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Metadata training stage failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Unified 3-Class Phishing Training Pipeline")
    parser.add_argument('--dev', action='store_true', help="Development mode: use small sample size")
    parser.add_argument('--skip-preprocess', action='store_true', help="Skip loading raw data, load cleaned_emails.csv instead")
    parser.add_argument('--skip-bert', action='store_true', help="Skip heavy BERT embedding generation")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for CPU embedding")
    args = parser.parse_args()

    processed_path = PROJECT_ROOT / 'data/processed/cleaned_emails.csv'

    # Create directories
    for p in ['data/processed/features', 'data/processed/embeddings', 'data/splits', 'models/ml/ensemble', 'models/dl', 'models/vectorizers']:
        os.makedirs(PROJECT_ROOT / p, exist_ok=True)

    # Stage 1: Load and Preprocess
    if args.skip_preprocess:
        logger.info("=== STAGE 1: Skipping Preprocessing, loading existing Cleaned CSV ===")
        if not processed_path.exists():
             logger.error(f"✗ Cannot skip preprocess: {processed_path} not found!")
             return
        df_clean = pd.read_csv(processed_path)
    else:
        logger.info("=== STAGE 1: Data Loading plus Preprocessing ===")
        df = load_all_datasets()
        if args.dev:
            logger.info("DEV MODE: Sampling 1000 rows only.")
            df = df.sample(min(1000, len(df)), random_state=42)
        
        df_clean = process_dataframe(df, text_column='raw_text', label_column='unified_label')
        df_clean.to_csv(processed_path, index=False)
        logger.info(f"✓ Preprocessed {len(df_clean)} emails saved to {processed_path}")

    # Stage 2: Metadata/URL Training (Requested Addition)
    run_stage_metadata_training()

    # Stage 3: Vectorization (TF-IDF)
    logger.info("=== STAGE 3: TF-IDF Vectorization ===")
    df_clean = df_clean.dropna(subset=['cleaned_text', 'unified_label']).reset_index(drop=True)
    labels = df_clean['unified_label'].values
    np.save(PROJECT_ROOT / 'data/splits/canonical_labels.npy', labels)

    train_idx, test_idx = train_test_split(np.arange(len(df_clean)), test_size=0.3, random_state=42, stratify=labels)
    np.save(PROJECT_ROOT / 'data/splits/canonical_train_idx.npy', train_idx)
    np.save(PROJECT_ROOT / 'data/splits/canonical_test_idx.npy', test_idx)

    logger.info("Fitting TF-IDF...")
    vec = fit_tfidf(df_clean['cleaned_text'].tolist())
    X_tfidf = transform_tfidf(vec, df_clean['cleaned_text'].tolist())
    save_tfidf_features(X_tfidf, PROJECT_ROOT / 'data/processed/features/tfidf_features.pkl')
    save_tfidf_vectorizer(vec, PROJECT_ROOT / 'models/vectorizers/tfidf_vectorizer.pkl')
    logger.info(f"✓ TF-IDF complete. Matrix shape: {X_tfidf.shape}")

    # Stage 4: BERT Embeddings
    embedding_path = PROJECT_ROOT / 'data/processed/embeddings/distilbert_embeddings.npy'
    if args.skip_bert:
        logger.info("=== STAGE 4: Skipping BERT Embeddings per --skip-bert flag ===")
    else:
        logger.info("=== STAGE 4: Generating DistilBERT Embeddings (CPU Optimized) ===")
        logger.info("Note: This stage is extremely slow on CPU. Please be patient.")
        embeddings = get_distilbert_embeddings(
            df_clean['cleaned_text'].tolist(),
            save_path=str(embedding_path),
            batch_size=args.batch_size
        )
        np.save(PROJECT_ROOT / 'data/processed/embeddings/distilbert_labels.npy', labels)
        logger.info(f"✓ BERT Embeddings complete. Shape: {embeddings.shape}")

    # Stage 5: ML Ensemble Training
    logger.info("=== STAGE 5: Training ML Ensemble ===")
    X_train, y_train = X_tfidf[train_idx], labels[train_idx]
    
    from src.utils.smote_balancer import apply_smote
    logger.info("Applying SMOTE balancing...")
    X_bal, y_bal = apply_smote(X_train, y_train)

    # SuperLearner expects a dictionary of base learners
    models_dict = {
        'LogisticRegression': build_logistic_regression(),
        'RandomForest': build_random_forest(),
        'SVM': build_svm(),
        'XGBoost': build_xgboost(),
    }

    logger.info("Training Super Learner ensemble...")
    # class_names is internal to SuperLearner, not an init arg
    sl = SuperLearner(base_learners=models_dict)
    sl.fit(X_bal, y_bal)
    
    # Save SL and individual models
    os.makedirs(PROJECT_ROOT / 'models/ml/ensemble', exist_ok=True)
    joblib.dump(sl, PROJECT_ROOT / 'models/ml/ensemble/super_learner.joblib')
    for name, model in sl._fitted_base.items():
        joblib.dump(model, PROJECT_ROOT / f'models/ml/{name.lower()}.joblib')
        
    logger.info("✓ Ensemble training complete.")

    # Stage 6: BiLSTM Training
    if os.path.exists(embedding_path):
        logger.info("=== STAGE 6: Training BiLSTM Deep Learning Layer ===")
        emb = np.load(embedding_path)
        labels_dl = np.load(PROJECT_ROOT / 'data/processed/embeddings/distilbert_labels.npy', allow_pickle=True)
        
        min_len = min(len(emb), len(labels_dl))
        train_idx_clipped = train_idx[train_idx < min_len]
        
        X_train_dl, y_train_dl = emb[train_idx_clipped], labels_dl[train_idx_clipped]
        label_to_idx = {'legitimate': 0, 'phishing_human': 1, 'phishing_ai': 2}
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_dl, y_train_dl, test_size=0.2, random_state=42)

        model, history = train_bilstm(X_tr, y_tr, X_val, y_val, label_to_idx)
        logger.info("✓ BiLSTM training complete.")
    else:
        logger.warning("Skipping BiLSTM: Embeddings not found.")

    logger.info("========================================")
    logger.info("🚀 FULL PIPELINE SYNC COMPLETE!")
    logger.info("========================================")

if __name__ == '__main__':
    main()
