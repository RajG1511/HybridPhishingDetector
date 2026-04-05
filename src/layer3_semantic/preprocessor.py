"""
Layer 3 — NLP Text Preprocessor

Cleans raw email body text for downstream vectorization:
  - Strips HTML tags, JavaScript, CSS, and special characters
  - Tokenizes using NLTK or SpaCy
  - Removes stop words
  - Applies lemmatization

Input:  raw email body string
Output: list of cleaned tokens or a single cleaned string
"""

import logging
import re
import string

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for resource in ("punkt", "stopwords", "wordnet"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else resource)
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


def strip_html(text: str) -> str:
    """Remove all HTML tags, scripts, and styles from a string.

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
    """Normalize whitespace and remove non-printable characters.

    Args:
        text: Input string (already HTML-stripped).

    Returns:
        Cleaned string with normalized whitespace.
    """
    text = re.sub(r"[^\x20-\x7E]", " ", text)  # Keep printable ASCII
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize and lemmatize cleaned text using NLTK.

    Args:
        text: Cleaned plain-text string.
        remove_stopwords: Whether to filter out common stop words.

    Returns:
        List of lemmatized, lowercase tokens.
    """
    _ensure_nltk()
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t.isalpha()]

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    logger.debug("Tokenized to %d tokens", len(tokens))
    return tokens


def preprocess(raw_body: str, return_tokens: bool = False) -> str | list[str]:
    """Full preprocessing pipeline for a raw email body.

    Args:
        raw_body: Raw email body (plain text or HTML).
        return_tokens: If True, return a list of tokens; otherwise return a
            single cleaned string.

    Returns:
        List of tokens or cleaned string depending on return_tokens.
    """
    text = strip_html(raw_body)
    text = clean_text(text)
    tokens = tokenize_and_lemmatize(text)

    if return_tokens:
        return tokens
    return " ".join(tokens)
