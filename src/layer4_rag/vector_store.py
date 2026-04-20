"""
Layer 4 — User Email History Vector Store

Maintains a vector database of a user's legitimate email history to provide
contextual baselines for grey-zone email triage. Enables semantic similarity
search to detect anomalous communication patterns.

Dependencies: chromadb (optional), numpy
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory or persistent vector store for user email embeddings.

    Falls back to an in-memory numpy-based store if chromadb is not installed.

    Args:
        persist_dir: Optional path to persist the vector database to disk.
        collection_name: Name of the collection within the store.
    """

    def __init__(self, persist_dir: Path | None = None, collection_name: str = "user_history") -> None:
        self.collection_name = collection_name
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict] = []
        self._use_chroma = False

        if persist_dir is not None:
            try:
                import chromadb  # type: ignore[import]

                self._client = chromadb.PersistentClient(path=str(persist_dir))
                self._collection = self._client.get_or_create_collection(collection_name)
                self._use_chroma = True
                logger.info("Using ChromaDB persistent store at %s", persist_dir)
            except ImportError:
                logger.warning("chromadb not installed; using in-memory fallback store")

    def add(self, embedding: np.ndarray, metadata: dict, doc_id: str) -> None:
        """Add an email embedding to the store.

        Args:
            embedding: 1-D float array representing the email.
            metadata: Arbitrary metadata dict (e.g. sender, date).
            doc_id: Unique identifier for the document.
        """
        if self._use_chroma:
            self._collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[doc_id],
            )
        else:
            self._embeddings.append(embedding)
            self._metadata.append({"id": doc_id, **metadata})

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Retrieve the most semantically similar stored emails.

        Args:
            embedding: Query embedding vector.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of metadata dicts for the top_k most similar stored emails.
        """
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k,
            )
            return results.get("metadatas", [[]])[0]

        if not self._embeddings:
            return []

        matrix = np.vstack(self._embeddings)
        # Cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        query_norm = np.linalg.norm(embedding) + 1e-10
        similarities = (matrix / norms) @ (embedding / query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self._metadata[i] for i in top_indices]
