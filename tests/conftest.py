"""
Test Configuration — Shared pytest Fixtures

Provides reusable fixtures for email parsing, feature extraction, and pipeline
tests. Fixtures use minimal synthetic data to avoid requiring downloaded datasets.
"""

import pytest


SAMPLE_RAW_EMAIL = """\
From: attacker@evil-bank.com
Reply-To: collect@phish.net
Return-Path: <bounce@evil-bank.com>
Subject: Urgent: Verify your account
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Dear Customer,

Your account has been suspended. Click here immediately to verify:
http://192.168.1.1/login?redirect=http://evil-bank.com/steal

Regards,
Security Team
"""

SAMPLE_HAM_EMAIL = """\
From: alice@company.com
Reply-To: alice@company.com
Return-Path: <alice@company.com>
Subject: Q3 Budget Review
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Hi Bob,

Please find attached the Q3 budget review document.

Best,
Alice
"""


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
