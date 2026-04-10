"""
Ebbinghaus Forgetting Curve Scorer.

Implements the cognitive science concept that memories decay over time
unless reinforced through recall. This is the unique "hook" of NeuroMem.

Formula: R = e^(-t/S)
Where:
    R = retention (0.0 → 1.0)
    t = time since creation (in days)
    S = stability factor = importance × (1 + 0.3 × recall_count)

A memory is a deletion candidate when R < threshold (default 0.2).
Recalled memories get an importance boost (spaced repetition effect).
"""
import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from app.models.memory import EpisodicMemory


class EbbinghausScorer:
    """
    Implements the Ebbinghaus Forgetting Curve for memory decay scoring.
    """

    DEFAULT_DECAY_THRESHOLD = 0.2

    def score(
        self,
        importance: float,
        days_since_created: float,
        recall_count: int,
    ) -> float:
        """
        Compute retention score for a memory.

        Formula: R = e^(-t / S)
        Where:
            t = time since creation in days
            S = 14 * (importance + 0.1) * (1 + 0.5 * recall_count)

        The 14-day base period means a medium-importance memory (0.5)
        with no recalls decays to ~50% after ~5.8 days.
        The +0.1 floor prevents near-zero importance from causing
        instant decay (division-by-near-zero).

        This formula matches the SQL bulk update in episodic/store.py
        to ensure consistent scoring across Python and database paths.

        Returns:
            Retention score 0.0 → 1.0
        """
        if importance < 0:
            importance = 0.0

        # Stability grows with importance and recall frequency
        # 14-day base period with importance floor of 0.1
        stability = 14.0 * (importance + 0.1) * (1 + 0.5 * recall_count)

        # Ebbinghaus: R = e^(-t/S)
        retention = math.exp(-days_since_created / stability)
        return max(0.0, min(1.0, retention))

    def score_memory(self, memory: EpisodicMemory, sim_now: Optional[datetime] = None) -> float:
        """Compute decay score for an EpisodicMemory object."""
        now = sim_now or datetime.now(timezone.utc)
        days = (now - memory.created_at).total_seconds() / 86400.0
        return self.score(
            importance=memory.importance,
            days_since_created=days,
            recall_count=memory.recall_count,
        )

    def score_batch(
        self, memories: List[EpisodicMemory]
    ) -> List[Tuple[EpisodicMemory, float]]:
        """Score a list of memories, returning (memory, score) pairs."""
        return [(m, self.score_memory(m)) for m in memories]

    def filter_for_deletion(
        self,
        memories: List[EpisodicMemory],
        threshold: float = DEFAULT_DECAY_THRESHOLD,
    ) -> Tuple[List[EpisodicMemory], List[EpisodicMemory]]:
        """
        Split memories into (keep, delete) lists based on decay threshold.

        Returns:
            (memories_to_keep, memories_to_delete)
        """
        keep, delete = [], []
        for memory, score in self.score_batch(memories):
            if score < threshold:
                delete.append(memory)
            else:
                keep.append(memory)
        return keep, delete

    def reinforcement_bonus(self, current_importance: float) -> float:
        """
        Bump importance when a memory is recalled (spaced repetition effect).
        Importance increases by 10%, capped at 1.0.
        """
        return min(1.0, current_importance * 1.10)
