"""
Token Budget Manager.

Ensures memory injection doesn't overflow the LLM's context window.
Over-fetches candidates, re-ranks with a cross-encoder, then greedily
fits the most relevant memories within a configurable token budget.

Production-critical: prevents context window overflow that causes
silent truncation or API errors in production deployments.
"""
import logging
from functools import lru_cache
from typing import List, Optional

import tiktoken
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.models.memory import RetrievedMemory

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Singleton cross-encoder instance (cached after first call)."""
    logger.info(
        "loading_reranker",
        extra={"model": settings.reranker_model},
    )
    return CrossEncoder(settings.reranker_model, max_length=512)


class TokenBudgetManager:
    """
    Token-budget-aware memory retrieval manager.

    Pipeline:
    1. Over-fetch 3x candidates from all tiers
    2. Cross-encoder re-rank for precision
    3. Greedy token-budgeted selection
    """

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens or settings.memory_token_budget
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base encoding)."""
        return len(self._encoder.encode(text))

    def rerank(
        self,
        query: str,
        memories: List[RetrievedMemory],
        top_k: int = 10,
    ) -> List[RetrievedMemory]:
        """
        Re-rank memories using a cross-encoder for precision.

        The cross-encoder scores (query, memory) pairs jointly,
        producing far more accurate relevance scores than
        bi-encoder cosine similarity alone.
        """
        if not memories:
            return []

        reranker = get_reranker()
        pairs = [(query, mem.content) for mem in memories]
        scores = reranker.predict(pairs)

        # Update relevance scores with cross-encoder scores
        scored_memories = []
        for mem, score in zip(memories, scores):
            reranked = mem.model_copy()
            reranked.relevance_score = float(score)
            scored_memories.append(reranked)

        # Sort by re-ranked score
        scored_memories.sort(key=lambda m: m.relevance_score, reverse=True)
        return scored_memories[:top_k]

    def fit_to_budget(
        self, memories: List[RetrievedMemory]
    ) -> List[RetrievedMemory]:
        """
        Greedily select memories that fit within the token budget.

        Adds memories in order of relevance until the budget is exhausted.
        This prevents context window overflow in production.
        """
        if not memories:
            return []

        fitted = []
        used_tokens = 0

        for mem in sorted(
            memories, key=lambda m: m.relevance_score, reverse=True
        ):
            tokens = self.count_tokens(mem.content)
            if used_tokens + tokens > self.max_tokens:
                if not fitted:
                    # Always include at least one memory
                    fitted.append(mem)
                break
            fitted.append(mem)
            used_tokens += tokens

        logger.info(
            "token_budget_fitted",
            extra={
                "total_candidates": len(memories),
                "fitted": len(fitted),
                "tokens_used": used_tokens,
                "budget": self.max_tokens,
            },
        )
        return fitted

    def rerank_and_fit(
        self,
        query: str,
        memories: List[RetrievedMemory],
        top_k: int = 10,
    ) -> List[RetrievedMemory]:
        """
        Full pipeline: re-rank with cross-encoder, then fit to token budget.
        """
        reranked = self.rerank(query, memories, top_k=top_k)
        return self.fit_to_budget(reranked)
