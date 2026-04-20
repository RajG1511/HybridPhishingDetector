"""
Layer 3 — Feature Vectorizer

Produces numeric feature vectors from preprocessed email text using two
complementary representations:
  1. TF-IDF sparse vectors (lexical/statistical signal) — implemented here
  2. DistilBERT dense embeddings (semantic/contextual signal) — deferred

Saved vectorizer artifacts are stored under models/vectorizers/.

Dependencies: scikit-learn, joblib; transformers + torch (for DistilBERT)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import DISTILBERT_MODEL_NAME, MODELS_DIR, TFIDF_MAX_FEATURES

logger = logging.getLogger(__name__)

_VECTORIZERS_DIR = MODELS_DIR / "vectorizers"
TFIDF_VECTORIZER_PATH = _VECTORIZERS_DIR / "tfidf_vectorizer.pkl"
TFIDF_FEATURES_PATH = MODELS_DIR.parent / "data" / "processed" / "features" / "tfidf_features.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF
# ══════════════════════════════════════════════════════════════════════════════

def fit_tfidf(corpus: list[str]) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a corpus of preprocessed email texts.

    Args:
        corpus: List of cleaned email body strings.

    Returns:
        Fitted TfidfVectorizer instance.
    """
    effective_min_df = 1 if len(corpus) < 5 else 2

    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=effective_min_df,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    vec.fit(corpus)
    logger.info(
        "TF-IDF fitted on %d documents (%d features, min_df=%d)",
        len(corpus),
        len(vec.vocabulary_),
        effective_min_df,
    )
    return vec


def transform_tfidf(vectorizer: TfidfVectorizer, texts: list[str]) -> np.ndarray:
    """Transform texts to a dense TF-IDF feature matrix.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        texts: List of preprocessed email strings.

    Returns:
        Dense numpy array of shape (n_samples, n_features).
    """
    sparse = vectorizer.transform(texts)
    return sparse.toarray()


def save_tfidf_vectorizer(vectorizer: TfidfVectorizer, path: Path | None = None) -> Path:
    """Serialize the fitted TF-IDF vectorizer to disk.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        path: Destination path. Defaults to models/vectorizers/tfidf_vectorizer.pkl.

    Returns:
        Path to the saved file.
    """
    dest = Path(path) if path else TFIDF_VECTORIZER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, dest)
    logger.info("Saved TF-IDF vectorizer to %s", dest)
    return dest


def load_tfidf_vectorizer(path: Path | None = None) -> TfidfVectorizer:
    """Load a serialized TF-IDF vectorizer from disk.

    Args:
        path: Source path. Defaults to models/vectorizers/tfidf_vectorizer.pkl.

    Returns:
        Loaded TfidfVectorizer.
    """
    src = Path(path) if path else TFIDF_VECTORIZER_PATH
    vec = joblib.load(src)
    logger.info("Loaded TF-IDF vectorizer from %s", src)
    return vec


def save_tfidf_features(matrix: np.ndarray, path: Path | None = None) -> Path:
    """Serialize the TF-IDF feature matrix to disk.

    Args:
        matrix: Dense numpy array of shape (n_samples, n_features).
        path: Destination path. Defaults to data/processed/features/tfidf_features.pkl.

    Returns:
        Path to the saved file.
    """
    dest = Path(path) if path else TFIDF_FEATURES_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(matrix, dest)
    logger.info("Saved TF-IDF feature matrix %s to %s", matrix.shape, dest)
    return dest


def load_tfidf_features(path: Path | None = None) -> np.ndarray:
    """Load a serialized TF-IDF feature matrix from disk.

    Args:
        path: Source path. Defaults to data/processed/features/tfidf_features.pkl.

    Returns:
        Dense numpy array.
    """
    src = Path(path) if path else TFIDF_FEATURES_PATH
    matrix = joblib.load(src)
    logger.info("Loaded TF-IDF feature matrix %s from %s", matrix.shape, src)
    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# DistilBERT embeddings (deferred — implement in Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

def get_distilbert_embeddings(
    texts: list[str],
    batch_size: int = 256,
    max_length: int = 128,
    show_progress: bool = True,
    save_path: Path | None = None,
) -> np.ndarray:
    """Compute DistilBERT [CLS] token embeddings for a list of texts.

    Extracts the [CLS] token hidden state (first token) from DistilBERT's
    last hidden layer as a 768-dim representation for each email. This is
    faster and more memory-efficient than mean-pooling over all tokens.

    Args:
        texts: List of preprocessed email strings.
        batch_size: Number of texts per GPU forward pass (default 256).
        max_length: Maximum token sequence length (default 128). Emails
            are typically short after preprocessing; 128 tokens captures
            >95% of content while being 4x faster than 512.
        show_progress: Whether to display a tqdm progress bar.
        save_path: Optional path to save the resulting .npy array immediately
            after generation (for checkpointing on large datasets).

    Returns:
        Array of shape (n_samples, 768).
    """
    import torch
    from transformers import DistilBertModel, DistilBertTokenizerFast

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)
    model = DistilBertModel.from_pretrained(DISTILBERT_MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("DistilBERT running on %s", device)

    all_embeddings: list[np.ndarray] = []
    n_batches = -(-len(texts) // batch_size)
    iterator = range(0, len(texts), batch_size)

    if show_progress and _tqdm is not None:
        iterator = _tqdm(iterator, total=n_batches, desc="DistilBERT embeddings", unit="batch")

    for i in iterator:
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # Enable Half-Precision (FP16) for A100 Tensor Cores
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                output = model(**encoded)

        # [CLS] token is the first token in every sequence
        cls_embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(cls_embeddings)

    result = np.vstack(all_embeddings)
    logger.info("DistilBERT embeddings complete: shape %s", result.shape)

    if save_path is not None:
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, result)
        logger.info("Saved embeddings to %s", dest)

    return result
