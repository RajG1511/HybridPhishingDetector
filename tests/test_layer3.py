"""
Unit Tests — Layer 3: Semantic Analysis Pipeline

Tests for text preprocessing, TF-IDF vectorization, and ensemble model
interface compliance.
"""

import numpy as np
import pytest

from src.layer3_semantic.preprocessor import strip_html, clean_text, preprocess


class TestPreprocessor:
    def test_strip_html_removes_tags(self) -> None:
        html = "<p>Hello <b>World</b></p><script>alert(1)</script>"
        result = strip_html(html)
        assert "<" not in result
        assert "Hello" in result
        assert "World" in result

    def test_clean_text_normalizes_whitespace(self) -> None:
        text = "hello   world\t\ntest"
        result = clean_text(text)
        assert "  " not in result
        assert result == "hello world test"

    def test_preprocess_returns_string(self) -> None:
        body = "<p>Click here to verify your account urgently!</p>"
        result = preprocess(body)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preprocess_returns_tokens(self) -> None:
        body = "Dear customer please verify your account"
        tokens = preprocess(body, return_tokens=True)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)


class TestVectorizer:
    def test_tfidf_fit_transform_shape(self) -> None:
        from src.layer3_semantic.vectorizer import fit_tfidf, transform_tfidf

        corpus = ["this is a phishing email", "hello how are you", "click here now"]
        vec = fit_tfidf(corpus)
        X = transform_tfidf(vec, corpus)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_tfidf_unseen_text(self) -> None:
        from src.layer3_semantic.vectorizer import fit_tfidf, transform_tfidf

        corpus = ["phishing email urgent", "legitimate business email"]
        vec = fit_tfidf(corpus)
        X = transform_tfidf(vec, ["new unseen text here"])
        assert X.shape[0] == 1
