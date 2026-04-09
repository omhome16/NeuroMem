"""
BGE-M3 Embedding Model.

Produces 1024-dimensional dense vectors for semantic similarity search.
Configurable via EMBEDDING_MODEL env var — can swap to lighter models
like all-MiniLM-L6-v2 (384-dim) for faster iteration.
"""
import logging
from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Embedder:
    """
    Dense embedding model for semantic memory vectors.
    Uses sentence-transformers under the hood.
    """

    def __init__(self):
        logger.info(
            "loading_embedding_model",
            extra={"model": settings.embedding_model, "device": settings.embedding_device},
        )
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        self.dim = settings.embedding_dim

    def embed(self, text: str) -> List[float]:
        """Embed a single string into a dense vector."""
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings efficiently in a single batch."""
        if not texts:
            return []
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
        return [e.tolist() for e in embeddings]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Singleton embedder instance (cached after first call)."""
    return Embedder()
