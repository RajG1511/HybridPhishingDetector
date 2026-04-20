# Layer 1 Handoff

This document is the handoff for Person 1's work on Layer 1: Email Protocol Authentication.

Layer 1 is ready to be consumed by downstream teammates. The implementation now supports:

- `.eml` parsing with Python's `email` library
- header extraction for `From`, `Reply-To`, `Return-Path`, `Message-ID`, `Received-SPF`, `Authentication-Results`, `DKIM-Signature`, and `ARC-Seal`
- header mismatch detection
- SPF status resolution from existing headers, plus best-effort DNS-backed SPF evaluation
- DKIM verification when `dkimpy` is installed
- ARC validation when `dkimpy` is installed
- a numeric `protocol_risk_score`
- unit and integration coverage for the Layer 1 handoff path

## Main Entry Point

Frontend and API integration should treat this as the Layer 1 entry point:

```python
from src.layer1_protocol import analyze_protocol_authentication
```

It returns a dictionary shaped like this:

```python
{
    "spf": "pass|fail|softfail|neutral|none|temperror|permerror|unknown",
    "dkim": "pass|fail|missing|unknown",
    "arc": "pass|fail|missing|unknown",
    "header_mismatch": bool,
    "header_issues": list[str],
    "metadata_flags": list[str],
    "protocol_risk_score": int,
    "metadata_features": dict,
}
```

## Expected Input

`analyze_protocol_authentication(...)` can work with:

- raw email `bytes`
- raw email `str`
- parsed email objects already normalized by the pipeline

For uploaded `.eml` files, the easiest path is:

```python
raw_bytes = uploaded_file_bytes
layer1_output = analyze_protocol_authentication(raw_bytes)
```

## Important Output Fields

These are the fields the frontend should care about first:

- `protocol_risk_score`
  This is the Layer 1 numeric score on a `0-100` scale.
- `spf`
  SPF result status.
- `dkim`
  DKIM result status.
- `arc`
  ARC result status.
- `header_mismatch`
  Fast boolean for whether sender-related headers look suspicious.
- `header_issues`
  Human-readable mismatch explanations.
- `metadata_flags`
  Human-readable Layer 1 risk signals that can be displayed in the UI.

## Useful Nested Fields

Inside `metadata_features`, these are especially useful for display or debugging:

- `from_domain`
- `reply_to_domain`
- `return_path_domain`
- `message_id_domain`
- `sender_ip`
- `from_reply_to_mismatch`
- `from_return_path_mismatch`
- `message_id_domain_mismatch`
- `num_received_headers`

## Recommended Frontend/API Assumptions

- Treat Layer 1 as a standalone module.
- Do not assume `spf`, `dkim`, or `arc` will always be `pass` or `fail`.
- Be ready for `missing` or `unknown` values.
- Use `header_issues` and `metadata_flags` directly for display.
- Use `protocol_risk_score` as the summary number for this layer.
- Do not hardcode score thresholds in the frontend unless the backend team agrees on them first.

## Demo Commands

These commands can be used to inspect the current Layer 1 output:

```bash
python scripts/run_layer1_demo.py --sample phishing
python scripts/run_layer1_demo.py --sample ham --json
python scripts/run_layer1_demo.py --file tests\fixtures\sample_ham.eml --json
```

## Test Commands

Run the Layer 1 validation suite with:

```bash
python -m pytest tests\test_layer1.py tests\test_metadata_features.py tests\test_pipeline.py -q
```

Current verified result:

```text
30 passed
```

## Relevant Files

- `src/layer1_protocol/__init__.py`
- `src/layer1_protocol/header_parser.py`
- `src/layer1_protocol/spf_checker.py`
- `src/layer1_protocol/dkim_verifier.py`
- `src/layer1_protocol/arc_validator.py`
- `src/layer1_protocol/metadata_features.py`
- `src/pipeline/email_ingester.py`
- `scripts/run_layer1_demo.py`
- `tests/test_layer1.py`
- `tests/test_metadata_features.py`
- `tests/test_pipeline.py`
- `tests/fixtures/`

## Known Notes

- If an email does not include authentication headers, Layer 1 may legitimately return missing protocol evidence.
- Ham emails are not guaranteed to produce all-pass results unless the `.eml` itself actually contains valid SPF/DKIM/ARC evidence.
- SPF is best-effort and supports common mechanisms, but it is not a full RFC-complete SPF engine.
- DKIM and ARC require the relevant dependencies and usable message/header material to validate successfully.

## Suggested Integration Pattern

For the teammate working on the API or UI later, the expected flow is:

1. Accept uploaded `.eml` bytes or pasted raw email content.
2. Run `analyze_protocol_authentication(...)`.
3. Store the full returned dict in the backend response.
4. Display:
   - Layer 1 status badges for SPF, DKIM, and ARC
   - the `protocol_risk_score`
   - any `header_issues`
   - any `metadata_flags`

## Short Version

If you only need the essentials:

- Call `analyze_protocol_authentication(raw_email_bytes)`
- Read `protocol_risk_score`, `spf`, `dkim`, `arc`, `header_issues`, and `metadata_flags`
- Use `python scripts/run_layer1_demo.py --sample phishing --json` to see the contract quickly
- Use `python -m pytest tests\test_layer1.py tests\test_metadata_features.py tests\test_pipeline.py -q` to verify the handoff
