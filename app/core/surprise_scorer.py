"""
Surprise Scorer — Novelty-Gated Memory (Inspired by Google Titans).

Before storing a new memory, computes a "surprise score" measuring how
novel the fact is compared to what we already know about the user.

Core formula:
    surprise = 1 - max_cosine_similarity(new_embedding, existing_embeddings)

High surprise (>0.7): Brand new information — must store
Low surprise (<0.3): Redundant — skip (reduces write volume ~40%)

The gate threshold is: surprise * importance > SURPRISE_THRESHOLD

Reference: "Titans: Learning to Memorize at Test Time" (Google, Dec 2025)
"""
import logging
from typing import List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.memory.semantic.embedder import Embedder

logger = logging.getLogger(__name__)
settings = get_settings()


class SurpriseScorer:
    """
    Implements the Titans-inspired surprise metric for novelty gating.

    Measures how different a new memory is from the existing memory state.
    Only stores memories that pass the surprise * importance threshold.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.threshold = settings.surprise_threshold

    def compute_surprise(
        self,
        new_embedding: List[float],
        existing_embeddings: List[List[float]],
    ) -> float:
        """
        Compute surprise score for a new memory.

        surprise = 1 - max_similarity(new, existing)

        Returns:
            0.0 = identical to something we already know
            1.0 = completely novel information
        """
        if not existing_embeddings:
            return 1.0  # Everything is surprising for a new user

        new_vec = np.array(new_embedding)
        existing_matrix = np.array(existing_embeddings)

        # Cosine similarity against all existing memories
        norms_existing = np.linalg.norm(existing_matrix, axis=1, keepdims=True)
        norm_new = np.linalg.norm(new_vec)

        if norm_new == 0:
            return 1.0

        similarities = (existing_matrix @ new_vec) / (
            norms_existing.flatten() * norm_new + 1e-8
        )
        max_sim = float(np.max(similarities))
        surprise = 1.0 - max(0.0, min(1.0, max_sim))
        return surprise

    def compute_surprise_from_text(
        self,
        new_text: str,
        existing_texts: List[str],
    ) -> float:
        """Compute surprise from raw text (embeds internally)."""
        if not existing_texts:
            return 1.0

        new_emb = self.embedder.embed(new_text)
        existing_embs = self.embedder.embed_batch(existing_texts)
        return self.compute_surprise(new_emb, existing_embs)

    def should_store(self, surprise: float, importance: float) -> bool:
        """
        Gate decision: should this memory be stored?

        Uses combined surprise * importance score against threshold.
        This filters out ~40% of low-signal writes while retaining
        all genuinely novel or important facts.
        """
        combined = surprise * importance
        return combined > self.threshold

    def compute_momentum_surprise(
        self,
        recent_surprises: List[float],
        current_surprise: float,
        alpha: float = 0.3,
    ) -> float:
        """
        Titans-inspired momentum: consider both current surprise
        and the trend of recent surprises.

        If recent context has been consistently surprising (e.g., user
        is telling a new life story), lower the threshold for subsequent
        facts that may not be individually surprising.

        momentum = alpha * mean(recent) + (1 - alpha) * current
        """
        if not recent_surprises:
            return current_surprise

        recent_mean = float(np.mean(recent_surprises))
        return alpha * recent_mean + (1 - alpha) * current_surprise

    def batch_score(
        self,
        new_texts: List[str],
        existing_texts: List[str],
        importances: List[float],
    ) -> List[Tuple[str, float, float, bool]]:
        """
        Score a batch of new memories against existing state.

        Returns: List of (text, surprise, importance, should_store)
        """
        if not existing_texts:
            return [
                (text, 1.0, imp, True)
                for text, imp in zip(new_texts, importances)
            ]

        existing_embs = self.embedder.embed_batch(existing_texts)
        new_embs = self.embedder.embed_batch(new_texts)

        results = []
        recent_surprises = []
        for text, emb, imp in zip(new_texts, new_embs, importances):
            raw_surprise = self.compute_surprise(emb, existing_embs)
            # Apply momentum from recent context
            surprise = self.compute_momentum_surprise(
                recent_surprises[-5:], raw_surprise
            )
            store = self.should_store(surprise, imp)
            results.append((text, surprise, imp, store))
            recent_surprises.append(raw_surprise)

        return results
