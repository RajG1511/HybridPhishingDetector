"""
Layer 4 — User Communication Profile

Models a baseline communication pattern for each user from their legitimate
email history. Used to detect anomalous senders, topics, or writing styles
that deviate from normal correspondence, a key signal for targeted spear-phishing.

Dependencies: numpy, vector_store
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class UserProfile:
    """Baseline communication profile for a single user.

    Tracks:
        - Known sender domains
        - Common topic clusters (via embedding centroids)
        - Average email embedding for cosine anomaly scoring

    Args:
        user_id: Unique identifier for the user.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.known_sender_domains: Counter = Counter()
        self.embeddings: list[np.ndarray] = []
        self.mean_embedding: np.ndarray | None = None

    def update(self, sender_domain: str, embedding: np.ndarray) -> None:
        """Add a legitimate email to the profile.

        Args:
            sender_domain: Domain of the email sender (e.g. "company.com").
            embedding: Dense embedding of the email body.
        """
        self.known_sender_domains[sender_domain] += 1
        self.embeddings.append(embedding)
        self.mean_embedding = np.mean(self.embeddings, axis=0)

    def anomaly_score(self, sender_domain: str, embedding: np.ndarray) -> float:
        """Compute a 0–1 anomaly score for an incoming email.

        Higher scores indicate the email is more unusual relative to the
        user's normal communication history.

        Args:
            sender_domain: Domain of the incoming email sender.
            embedding: Dense embedding of the email body.

        Returns:
            Float anomaly score in [0.0, 1.0].
        """
        score = 0.0

        # Unknown sender domain is a weak signal
        if sender_domain not in self.known_sender_domains:
            score += 0.3

        # Cosine distance from mean embedding
        if self.mean_embedding is not None and len(self.embeddings) >= 5:
            cosine_sim = float(
                np.dot(embedding, self.mean_embedding)
                / (np.linalg.norm(embedding) * np.linalg.norm(self.mean_embedding) + 1e-10)
            )
            score += (1.0 - cosine_sim) * 0.7

        return min(score, 1.0)

    def is_known_sender(self, domain: str) -> bool:
        """Check if a sender domain has been seen before.

        Args:
            domain: Sender domain to check.

        Returns:
            True if the domain appears in this user's history.
        """
        return domain in self.known_sender_domains
