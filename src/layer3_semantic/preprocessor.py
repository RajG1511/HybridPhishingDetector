"""
Layer 3 — NLP Text Preprocessor

Cleans raw email body text for downstream vectorization:
  - Strips HTML tags, JavaScript, CSS, and special characters
  - Lowercases all text
  - Removes special characters and extra whitespace with regex
  - Tokenizes using NLTK word_tokenize
  - Removes English stop words
  - Lemmatizes with WordNetLemmatizer

Primary public API:
    preprocess_email(raw_body: str) -> str
    process_dataframe(df, text_column, label_column) -> pd.DataFrame
"""

import logging
import re
import string
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_NLTK_READY = False
# Module-level cached NLTK objects — avoids re-instantiation on every call
_LEMMATIZER = None
_STOP_WORDS = None

# Truncate email bodies to this many chars before tokenizing (perf guard)
_MAX_TEXT_CHARS = 4000


def _ensure_nltk() -> None:
    global _NLTK_READY, _LEMMATIZER, _STOP_WORDS
    if _NLTK_READY:
        return
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    needed = {
        "tokenizers/punkt_tab": "punkt_tab",
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for find_path, name in needed.items():
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(name, quiet=True)

    _LEMMATIZER = WordNetLemmatizer()
    _STOP_WORDS = set(stopwords.words("english"))
    _NLTK_READY = True


def strip_html(text: str) -> str:
    """Remove all HTML tags, scripts, styles, and head elements.

    Args:
        text: Raw HTML or mixed content string.

    Returns:
        Plain text with HTML removed.
    """
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "head", "meta"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """Lowercase, remove non-printable chars, and normalize whitespace.

    Args:
        text: Input string (HTML already stripped).

    Returns:
        Lowercase cleaned string with normalized whitespace.
    """
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Keep only printable ASCII
    text = re.sub(r"[^\x20-\x7e]", " ", text)
    # Remove special characters (keep letters, digits, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize and lemmatize cleaned text.

    Uses a fast regex tokenizer (avoids NLTK punkt overhead) and the
    module-level cached WordNetLemmatizer and stop-word set.

    Args:
        text: Cleaned lowercase plain-text string.
        remove_stopwords: Whether to filter out common English stop words.

    Returns:
        List of lemmatized tokens (alphabetic only, len >= 2).
    """
    _ensure_nltk()

    tokens = _TOKEN_RE.findall(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens


def preprocess_email(raw_body: str) -> str:
    """Full preprocessing pipeline for a raw email body.

    Strips HTML → lowercases → removes special chars/URLs → tokenizes →
    removes stop words → lemmatizes → rejoins as a single string.

    Args:
        raw_body: Raw email body (plain text or HTML).

    Returns:
        Single cleaned string of space-joined lemmatized tokens.
    """
    if not raw_body or not isinstance(raw_body, str):
        return ""
    try:
        # Truncate very long bodies before heavy processing
        raw_body = raw_body[:_MAX_TEXT_CHARS]
        text = strip_html(raw_body)
        text = clean_text(text)
        tokens = tokenize_and_lemmatize(text)
        return " ".join(tokens)
    except Exception as exc:
        logger.warning("preprocess_email failed: %s", exc)
        return ""


def process_dataframe(
    df,
    text_column: str,
    label_column: Optional[str] = None,
    batch_size: int = 1000,
) -> "pd.DataFrame":
    """Batch-preprocess a DataFrame of emails.

    Applies preprocess_email() to every row in text_column and appends the
    result as a new 'cleaned_text' column.

    Args:
        df: Input pandas DataFrame.
        text_column: Column name containing raw email body text.
        label_column: Optional label column to carry through (no transformation).
        batch_size: Log progress every N rows.

    Returns:
        New DataFrame with columns:
            - all original columns
            - cleaned_text (str): preprocessed email body
    """
    import pandas as pd

    result = df.copy()
    total = len(result)
    cleaned: list[str] = []

    for i, text in enumerate(result[text_column].fillna("").astype(str)):
        cleaned.append(preprocess_email(text))
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            logger.info("Preprocessed %d / %d rows", i + 1, total)

    result["cleaned_text"] = cleaned
    return result


# ── Legacy alias kept for internal use ──
def preprocess(raw_body: str, return_tokens: bool = False):
    """Alias for preprocess_email with optional token-list output.

    Args:
        raw_body: Raw email body.
        return_tokens: If True, return list of tokens instead of string.

    Returns:
        Cleaned string or list of tokens.
    """
    if not raw_body or not isinstance(raw_body, str):
        return [] if return_tokens else ""
    text = strip_html(raw_body)
    text = clean_text(text)
    tokens = tokenize_and_lemmatize(text)
    return tokens if return_tokens else " ".join(tokens)
