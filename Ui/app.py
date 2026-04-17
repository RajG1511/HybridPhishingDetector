from __future__ import annotations

import os
from typing import Any

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize backend response into a template-friendly structure."""
    layer3_proba = _safe_float(data.get("layer3_proba", 0.0))
    layer3_label = str(data.get("layer3_label", "unknown"))

    if layer3_proba > 0.0:
        confidence_display = f"{round(layer3_proba * 100, 2)}%"
    else:
        confidence_display = "N/A"

    layer4_used = bool(data.get("layer4_used", False))
    layer4_eligible = bool(data.get("layer4_eligible", False))
    layer4_reason = str(data.get("layer4_reason", "") or "")
    layer4_note = str(data.get("layer4_note", "") or "")

    return {
        "verdict": str(data.get("verdict", "unknown")),
        "risk_score": _safe_int(data.get("risk_score", 0)),
        "confidence": confidence_display,
        "summary": str(data.get("narrative", "No summary available.")),

        # Layer 1
        "layer1_spf": str(data.get("layer1_spf", "unknown")),
        "layer1_dkim": str(data.get("layer1_dkim", "unknown")),
        "layer1_arc": str(data.get("layer1_arc", "unknown")),
        "layer1_header_mismatch": bool(data.get("layer1_header_mismatch", False)),
        "layer1_protocol_risk_score": _safe_int(data.get("layer1_protocol_risk_score", 0)),
        "layer1_header_issues": list(data.get("layer1_header_issues", [])),
        "layer1_metadata_flags": list(data.get("layer1_metadata_flags", [])),

        # Layer 2
        "layer2_flags": list(data.get("layer2_flags", [])),
        "layer2_url_count": _safe_int(data.get("layer2_url_count", 0)),
        "layer2_feature_summary": dict(data.get("layer2_feature_summary", {})),

        # Layer 3
        "layer3_label": layer3_label,
        "layer3_proba": layer3_proba,

        # Layer 4
        "layer4_used": layer4_used,
        "layer4_eligible": layer4_eligible,
        "layer4_reason": layer4_reason,
        "layer4_note": layer4_note,

        # Explainability
        "shap_features": list(data.get("shap_features", [])),
        "lime_words": list(data.get("lime_words", [])),

        "raw_response": data,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        uploaded_file = request.files.get("eml_file")

        try:
            if uploaded_file and uploaded_file.filename:
                files = {
                    "file": (
                        uploaded_file.filename,
                        uploaded_file.stream,
                        uploaded_file.mimetype or "message/rfc822",
                    )
                }
                response = requests.post(
                    f"{API_BASE_URL}/analyze/upload",
                    files=files,
                    timeout=120,
                )
            elif email_text:
                submitted_text = email_text
                payload = {"email_text": email_text}
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json=payload,
                    timeout=120,
                )
            else:
                error = "Please paste an email or upload a .eml file."
                return render_template(
                    "index.html",
                    result=result,
                    error=error,
                    submitted_text=submitted_text,
                )

            if response.ok:
                result = normalize_result(response.json())
            else:
                try:
                    backend_error = response.json()
                except Exception:
                    backend_error = response.text
                error = f"Backend error ({response.status_code}): {backend_error}"

        except requests.exceptions.ConnectionError:
            error = (
                "Could not connect to the FastAPI backend. "
                "Make sure uvicorn api.main:app is running on port 8000."
            )
        except requests.exceptions.Timeout:
            error = "The backend request timed out."
        except Exception as exc:
            error = f"Unexpected error: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        submitted_text=submitted_text,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)