"""
Test Configuration — Shared pytest Fixtures

Provides reusable fixtures for email parsing, feature extraction, and pipeline
tests. Fixtures use minimal synthetic data to avoid requiring downloaded datasets.
"""

from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RAW_EMAIL = (FIXTURES_DIR / "sample_phishing.eml").read_text(encoding="utf-8")
SAMPLE_HAM_EMAIL = (FIXTURES_DIR / "sample_ham.eml").read_text(encoding="utf-8")
SAMPLE_HTML_ONLY_EMAIL = (FIXTURES_DIR / "sample_html_only.eml").read_text(encoding="utf-8")


@pytest.fixture
def phishing_eml_path() -> Path:
    """Return the path to the phishing .eml fixture."""
    return FIXTURES_DIR / "sample_phishing.eml"


@pytest.fixture
def ham_eml_path() -> Path:
    """Return the path to the legitimate .eml fixture."""
    return FIXTURES_DIR / "sample_ham.eml"


@pytest.fixture
def html_only_eml_path() -> Path:
    """Return the path to the HTML-only .eml fixture."""
    return FIXTURES_DIR / "sample_html_only.eml"


@pytest.fixture
def raw_phishing_email() -> bytes:
    """Return a synthetic phishing email as bytes."""
    return SAMPLE_RAW_EMAIL.encode("utf-8")


@pytest.fixture
def raw_ham_email() -> bytes:
    """Return a synthetic legitimate email as bytes."""
    return SAMPLE_HAM_EMAIL.encode("utf-8")


@pytest.fixture
def phishing_email_text() -> str:
    """Return a synthetic phishing email as a string."""
    return SAMPLE_RAW_EMAIL


@pytest.fixture
def ham_email_text() -> str:
    """Return a synthetic legitimate email as a string."""
    return SAMPLE_HAM_EMAIL


@pytest.fixture
def html_only_email_text() -> str:
    """Return an HTML-only email fixture as a string."""
    return SAMPLE_HTML_ONLY_EMAIL
