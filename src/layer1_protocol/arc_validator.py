"""
Layer 1 — ARC Chain Validator

Validates the Authenticated Received Chain (ARC) headers in forwarded emails.
ARC preserves authentication results across mail forwarding hops; a broken
chain indicates potential tampering or misconfiguration.

Dependencies: dkimpy (provides arc module), dnspython
"""

import logging

logger = logging.getLogger(__name__)


def validate_arc(raw_email: bytes) -> dict[str, bool | str | None]:
    """Validate the ARC chain in a raw email message.

    Args:
        raw_email: Raw email bytes including all headers.

    Returns:
        Dictionary with keys:
            - has_arc (bool): Whether ARC headers are present.
            - chain_valid (bool | None): True if chain validates, False if broken,
              None if no ARC headers or validation could not be performed.
            - cv (str | None): ARC chain validation result string (none/fail/pass).
    """
    result: dict[str, bool | str | None] = {
        "has_arc": False,
        "chain_valid": None,
        "cv": None,
    }

    has_arc = b"ARC-Seal:" in raw_email or b"arc-seal:" in raw_email.lower()
    result["has_arc"] = has_arc

    if not has_arc:
        return result

    try:
        import dkim  # type: ignore[import]

        cv, _, _ = dkim.arc_verify(raw_email)
        cv_str = cv.decode("utf-8") if isinstance(cv, bytes) else str(cv)
        result["cv"] = cv_str
        result["chain_valid"] = cv_str.lower() == "pass"
        logger.debug("ARC chain validation result: %s", cv_str)
    except ImportError:
        logger.error("dkimpy is not installed; ARC validation skipped")
    except Exception as exc:
        logger.warning("ARC validation error: %s", exc)

    return result
