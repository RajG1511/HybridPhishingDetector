"""
Layer 1 — DKIM Signature Verifier

Verifies DKIM signatures in raw email messages using the dkimpy library.
A failed or missing DKIM signature is a strong spoofing indicator.

Dependencies: dkimpy, dnspython
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def verify_dkim(raw_email: bytes) -> bool:
    """Verify the DKIM signature of a raw email message.

    Args:
        raw_email: Raw email bytes including headers and body.

    Returns:
        True if a valid DKIM signature is present, False otherwise.
    """
    try:
        import dkim  # type: ignore[import]

        result = dkim.verify(raw_email)
        logger.debug("DKIM verification result: %s", result)
        return bool(result)
    except ImportError:
        logger.error("dkimpy is not installed; DKIM verification skipped")
        return False
    except Exception as exc:
        logger.warning("DKIM verification error: %s", exc)
        return False


def has_dkim_header(raw_email: bytes) -> bool:
    """Check whether the email contains a DKIM-Signature header at all.

    Args:
        raw_email: Raw email bytes.

    Returns:
        True if a DKIM-Signature header is present.
    """
    return b"DKIM-Signature:" in raw_email or b"dkim-signature:" in raw_email.lower()
