"""
Layer 4 — RAG Contextual Profiling Engine

Handles grey-zone emails (risk score between GREY_ZONE_LOW and GREY_ZONE_HIGH)
by retrieving similar historical emails from the user's vector store and
passing them as context to a lightweight LLM for a final determination.

Dependencies: requests (for LLM API calls), vector_store
"""

import logging
import os

import numpy as np

from src.layer4_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine for grey-zone email triage.

    Args:
        vector_store: Populated VectorStore containing user email history.
        top_k: Number of similar historical emails to retrieve per query.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def analyze(self, email_text: str, email_embedding: np.ndarray) -> dict[str, str | float]:
        """Analyze a grey-zone email using RAG reasoning.

        Retrieves similar historical emails and calls the configured LLM
        endpoint to produce a final classification with rationale.

        Args:
            email_text: Raw or preprocessed email body text.
            email_embedding: Dense embedding vector for the email.

        Returns:
            Dictionary with keys:
                - decision (str): "phishing" or "benign".
                - confidence (float): 0.0–1.0 confidence in decision.
                - rationale (str): Human-readable explanation.
        """
        similar = self.vector_store.query(email_embedding, top_k=self.top_k)
        context_snippets = [m.get("snippet", "") for m in similar if m.get("snippet")]

        prompt = self._build_prompt(email_text, context_snippets)
        response = self._call_llm(prompt)

        return response

    def _build_prompt(self, email_text: str, context: list[str]) -> str:
        """Construct the LLM prompt with retrieved context.

        Args:
            email_text: The email to classify.
            context: List of similar historical email snippets.

        Returns:
            Formatted prompt string.
        """
        context_block = "\n---\n".join(context) if context else "No historical context available."
        return (
            "You are a cybersecurity analyst evaluating a potentially phishing email.\n\n"
            f"HISTORICAL CONTEXT (similar past emails from this user's history):\n{context_block}\n\n"
            f"EMAIL TO EVALUATE:\n{email_text}\n\n"
            "Based on the historical context and email content, determine whether this email is "
            "'phishing' or 'benign'. Respond with JSON: "
            '{"decision": "phishing|benign", "confidence": 0.0-1.0, "rationale": "..."}'
        )

    def _call_llm(self, prompt: str) -> dict[str, str | float]:
        """Call the configured LLM API endpoint.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Parsed LLM response as a dictionary.
        """
        import json
        import requests

        endpoint = os.getenv("LLM_API_ENDPOINT", "")
        model = os.getenv("LLM_MODEL_NAME", "llama4-scout")

        if not endpoint:
            logger.warning("LLM_API_ENDPOINT not configured; returning neutral RAG result")
            return {"decision": "benign", "confidence": 0.5, "rationale": "RAG layer not configured."}

        try:
            resp = requests.post(
                endpoint,
                json={"model": model, "prompt": prompt, "max_tokens": 256},
                timeout=10,
            )
            resp.raise_for_status()
            content = resp.json().get("text", "{}")
            return json.loads(content)
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            return {"decision": "benign", "confidence": 0.5, "rationale": f"LLM error: {exc}"}
