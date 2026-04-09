"""
Semantic Contradiction Detector.

Two-phase pipeline to detect when a new memory contradicts an existing one:
  Phase 1: Fast embedding similarity to find same-topic memories
  Phase 2: LLM verification to confirm genuine contradiction

Inspired by Zep's temporal fact invalidation approach.
When a contradiction is confirmed, the old memory gets a valid_to timestamp
instead of being deleted — maintaining full provenance.
"""
import logging
from typing import List, Optional
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.memory.semantic.embedder import Embedder

logger = logging.getLogger(__name__)
settings = get_settings()


class Contradiction(BaseModel):
    """A detected contradiction between two memories."""
    old_content: str
    new_content: str
    old_memory_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    resolution: str = "newer_wins"  # newer_wins | keep_both | merge


class ContradictionVerification(BaseModel):
    """LLM response for contradiction verification."""
    is_contradiction: bool = Field(description="Whether the two statements genuinely contradict each other")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident the model is")
    explanation: str = Field(description="Brief explanation of why it is or isn't a contradiction")


CONTRADICTION_SYSTEM = """You are a contradiction detection engine.
Given two statements about the same user, determine if they genuinely contradict each other.

A contradiction means the two statements CANNOT both be true at the same time.
Examples:
- "User lives in Mumbai" vs "User moved to Pune" → CONTRADICTION (can't live in both)
- "User is vegetarian" vs "User had chicken biryani" → CONTRADICTION
- "User likes Python" vs "User is learning Rust" → NOT a contradiction (can like both)
- "User works at Google" vs "User is a software engineer" → NOT a contradiction (complementary)"""


class ContradictionDetector:
    """
    Detects contradictions between memories using a two-phase pipeline:
    1. Embedding similarity to find same-topic candidates
    2. LLM verification to confirm genuine contradictions
    """

    def __init__(self, embedder: Embedder, llm_client: LLMClient):
        self.embedder = embedder
        self.llm = llm_client
        self.topic_similarity_threshold = 0.55

    async def detect(
        self,
        new_content: str,
        existing_contents: List[str],
        existing_ids: Optional[List[str]] = None,
    ) -> List[Contradiction]:
        """
        Detect contradictions between a new memory and existing memories.

        Phase 1: Find same-topic memories via embedding similarity
        Phase 2: LLM-verify each candidate as a genuine contradiction
        """
        if not existing_contents:
            return []

        # Phase 1: Fast embedding filter — find semantically similar (same topic)
        candidates = self._find_same_topic(new_content, existing_contents)

        if not candidates:
            return []

        logger.info(
            "contradiction_candidates_found",
            extra={"count": len(candidates)},
        )

        # Phase 2: LLM verification
        contradictions = []
        for idx, similarity in candidates:
            verification = await self._llm_verify(
                new_content, existing_contents[idx]
            )
            if verification and verification.is_contradiction:
                contradiction = Contradiction(
                    old_content=existing_contents[idx],
                    new_content=new_content,
                    old_memory_id=existing_ids[idx] if existing_ids else None,
                    confidence=verification.confidence,
                )
                contradictions.append(contradiction)
                logger.info(
                    "contradiction_detected",
                    extra={
                        "old": existing_contents[idx][:60],
                        "new": new_content[:60],
                        "confidence": verification.confidence,
                    },
                )

        return contradictions

    def _find_same_topic(
        self, new_content: str, existing_contents: List[str]
    ) -> List[tuple]:
        """
        Phase 1: Find memories about the same topic using embedding similarity.
        Returns list of (index, similarity) tuples above threshold.
        """
        new_emb = np.array(self.embedder.embed(new_content))
        existing_embs = np.array(self.embedder.embed_batch(existing_contents))

        norm_new = np.linalg.norm(new_emb)
        norms_existing = np.linalg.norm(existing_embs, axis=1)

        if norm_new == 0:
            return []

        similarities = (existing_embs @ new_emb) / (
            norms_existing * norm_new + 1e-8
        )

        candidates = [
            (i, float(sim))
            for i, sim in enumerate(similarities)
            if sim > self.topic_similarity_threshold
        ]
        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:5]  # Check top 5 at most

    async def _llm_verify(
        self, new_content: str, old_content: str
    ) -> Optional[ContradictionVerification]:
        """Phase 2: Use LLM to verify if two same-topic memories contradict."""
        prompt = f"""Statement A (EXISTING): {old_content}
Statement B (NEW): {new_content}

Do these two statements about the same user genuinely contradict each other?"""

        result = await self.llm.structured_output(
            user_content=prompt,
            output_schema=ContradictionVerification,
            system=CONTRADICTION_SYSTEM,
            temperature=0.0,
        )
        return result
