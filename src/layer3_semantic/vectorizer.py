"""
Layer 3 — Feature Vectorizer

Produces numeric feature vectors from preprocessed email text using two
complementary representations:
  1. TF-IDF sparse vectors (lexical/statistical signal)
  2. DistilBERT dense embeddings (semantic/contextual signal)

Saved vectorizer artifacts are stored under models/vectorizers/.

Dependencies: scikit-learn, transformers, torch
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import DISTILBERT_MODEL_NAME, MODELS_DIR, TFIDF_MAX_FEATURES

logger = logging.getLogger(__name__)

_VECTORIZERS_DIR = MODELS_DIR / "vectorizers"


def fit_tfidf(corpus: list[str]) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a corpus of preprocessed email texts.

    Args:
        corpus: List of cleaned email body strings.

    Returns:
        Fitted TfidfVectorizer instance.
    """
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    vec.fit(corpus)
    logger.info("TF-IDF fitted on %d documents", len(corpus))
    return vec


def transform_tfidf(vectorizer: TfidfVectorizer, texts: list[str]) -> np.ndarray:
    """Transform texts to TF-IDF feature matrix.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        texts: List of preprocessed email strings.

    Returns:
        Dense numpy array of shape (n_samples, max_features).
    """
    sparse = vectorizer.transform(texts)
    return sparse.toarray()


def get_distilbert_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Compute mean-pooled DistilBERT embeddings for a list of texts.

    Args:
        texts: List of raw or preprocessed email strings.
        batch_size: Number of texts per forward pass.

    Returns:
        Array of shape (n_samples, 768) — DistilBERT hidden size.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(DISTILBERT_MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # Mean pool over token dimension
        hidden = output.last_hidden_state  # (batch, seq_len, 768)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())

        logger.debug("Embedded batch %d/%d", i // batch_size + 1, -(-len(texts) // batch_size))

    return np.vstack(all_embeddings)
