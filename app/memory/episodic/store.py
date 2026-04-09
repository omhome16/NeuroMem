"""
Episodic Memory Store — PostgreSQL-backed (Tier 2).

Stores LLM-extracted facts with a 30-day TTL. Uses Redis for retrieval caching.
Conflict resolution: simple timestamp-based overwrite (newer wins).
"""
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import asyncpg
import redis.asyncio as aioredis

from app.config import get_settings
from app.models.memory import (
    EpisodicMemory,
    ExtractedMemory,
    MemoryType,
    RetrievedMemory,
    MemoryTier,
)
from app.memory.episodic.scorer import EbbinghausScorer

logger = logging.getLogger(__name__)
settings = get_settings()


class EpisodicMemoryStore:
    """
    PostgreSQL-backed episodic memory store with Redis retrieval cache.

    Content is stored as plaintext (simplified from AES-256 blueprint).
    Cache key: episodic_cache:{user_id}:{query_hash}
    Cache TTL: 300 seconds
    """

    CACHE_TTL = 300  # 5 minutes
    CACHE_KEY_TEMPLATE = "episodic_cache:{user_id}:{query_hash}"

    def __init__(
        self,
        pg_pool: asyncpg.Pool,
        redis_client: aioredis.Redis,
    ):
        self.pg = pg_pool
        self.redis = redis_client
        self.scorer = EbbinghausScorer()

    async def store_memories(
        self,
        user_id: str,
        extracted: List[ExtractedMemory],
        session_id: UUID,
        turn_index: int,
    ) -> List[EpisodicMemory]:
        """
        Store extracted memories with simple timestamp-based conflict resolution.

        For each new memory, check if a similar one exists (same type, similar content).
        If found, the newer one wins — mark the old one as expired (decay_score = 0).
        """
        stored = []
        for ext_mem in extracted:
            # Simple conflict check: look for existing memories of same type
            existing = await self._get_recent_by_type(user_id, ext_mem.memory_type)

            # Check for near-duplicate content (simple substring match)
            for existing_mem in existing:
                # If content is very similar, expire the old one (newer wins)
                if self._is_similar(ext_mem.content, existing_mem.content):
                    await self._expire_memory(existing_mem.id)
                    logger.info(
                        "conflict_resolved_newer_wins",
                        extra={
                            "old": existing_mem.content[:50],
                            "new": ext_mem.content[:50],
                        },
                    )

            # Store the new memory
            mem = await self._insert_memory(user_id, ext_mem, session_id, turn_index)
            stored.append(mem)

        # Invalidate retrieval cache for this user
        await self._invalidate_cache(user_id)
        return stored

    def _is_similar(self, new_content: str, old_content: str) -> bool:
        """
        Simple similarity check between two memory contents.
        Returns True if they discuss the same topic (crude but effective).
        """
        new_words = set(new_content.lower().split())
        old_words = set(old_content.lower().split())
        if not new_words or not old_words:
            return False
        overlap = len(new_words & old_words) / min(len(new_words), len(old_words))
        return overlap > 0.6  # 60% word overlap = probably same topic

    async def _insert_memory(
        self,
        user_id: str,
        extracted: ExtractedMemory,
        session_id: UUID,
        turn_index: int,
        created_at: Optional[datetime] = None,
    ) -> EpisodicMemory:
        """Insert a single memory record."""
        now = created_at or datetime.now(timezone.utc)
        expires_at = now + timedelta(days=30)
        row = await self.pg.fetchrow(
            """
            INSERT INTO episodic_memories
                (user_id, content, memory_type, importance,
                 recall_count, session_id, source_turn, created_at, expires_at, decay_score)
            VALUES ($1, $2, $3, $4, 0, $5, $6, $7, $8, 1.0)
            RETURNING *
            """,
            user_id,
            extracted.content,
            extracted.memory_type.value,
            extracted.importance,
            session_id,
            turn_index,
            now,
            expires_at,
        )

        return EpisodicMemory(
            id=row["id"],
            user_id=user_id,
            content=extracted.content,
            memory_type=extracted.memory_type,
            importance=row["importance"],
            recall_count=0,
            tags=[],
            source_turn=turn_index,
            session_id=session_id,
            created_at=row["created_at"],
            last_recalled=None,
            consolidated=False,
            decay_score=1.0,
        )

    async def retrieve(
        self,
        user_id: str,
        query_text: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        sim_now: Optional[datetime] = None,
    ) -> List[RetrievedMemory]:
        """
        Retrieve episodic memories for a user, sorted by recency and importance.
        Uses Redis cache for repeated queries.
        """
        # Check cache
        cache_key = self.CACHE_KEY_TEMPLATE.format(
            user_id=user_id,
            query_hash=hash(query_text + str(memory_types)) & 0xFFFFFFFF,
        )
        cached = await self.redis.get(cache_key)
        if cached:
            raw = json.loads(cached)
            return [RetrievedMemory(**m) for m in raw]

        # Build query
        now = sim_now or datetime.now(timezone.utc)
        type_filter = ""
        params: list = [user_id, now, limit]
        if memory_types:
            placeholders = ", ".join(f"${i+4}" for i in range(len(memory_types)))
            type_filter = f"AND memory_type IN ({placeholders})"
            params.extend([t.value for t in memory_types])

        rows = await self.pg.fetch(
            f"""
            SELECT * FROM episodic_memories
            WHERE user_id = $1
              AND consolidated = FALSE
              AND expires_at > $2
              AND decay_score > 0.05
              {type_filter}
            ORDER BY (importance * decay_score) DESC, created_at DESC
            LIMIT $3
            """,
            *params,
        )

        results = []
        for row in rows:
            results.append(
                RetrievedMemory(
                    content=row["content"],
                    memory_type=MemoryType(row["memory_type"]),
                    tier=MemoryTier.EPISODIC,
                    relevance_score=float(row["importance"]) * float(row["decay_score"]),
                    importance=float(row["importance"]),
                    created_at=row["created_at"],
                )
            )
            # Increment recall count (spaced repetition reinforcement)
            await self._increment_recall(row["id"], float(row["importance"]))

        # Cache results
        await self.redis.setex(
            cache_key,
            self.CACHE_TTL,
            json.dumps([r.model_dump(mode="json") for r in results]),
        )
        return results

    async def _get_recent_by_type(
        self, user_id: str, memory_type: MemoryType, limit: int = 5, sim_now: Optional[datetime] = None
    ) -> List[EpisodicMemory]:
        """Fetch recent memories of same type for conflict checking."""
        now = sim_now or datetime.now(timezone.utc)
        rows = await self.pg.fetch(
            """
            SELECT * FROM episodic_memories
            WHERE user_id = $1 AND memory_type = $2
              AND consolidated = FALSE AND expires_at > $3
              AND decay_score > 0.1
            ORDER BY created_at DESC LIMIT $4
            """,
            user_id, memory_type.value, now, limit,
        )
        return [
            EpisodicMemory(
                id=row["id"],
                user_id=user_id,
                content=row["content"],
                memory_type=MemoryType(row["memory_type"]),
                importance=float(row["importance"]),
                recall_count=int(row["recall_count"]),
                tags=list(row["tags"] or []),
                source_turn=row["source_turn"],
                session_id=row["session_id"],
                created_at=row["created_at"],
                last_recalled=row["last_recalled"],
                consolidated=row["consolidated"],
                decay_score=float(row["decay_score"]),
            )
            for row in rows
        ]

    async def _expire_memory(self, memory_id: UUID) -> None:
        """Mark a memory as expired (decay_score = 0)."""
        await self.pg.execute(
            "UPDATE episodic_memories SET decay_score = 0.0 WHERE id = $1",
            memory_id,
        )

    async def _increment_recall(
        self, memory_id: UUID, current_importance: float, sim_now: Optional[datetime] = None
    ) -> None:
        """Increment recall count and boost importance (spaced repetition)."""
        now = sim_now or datetime.now(timezone.utc)
        new_importance = min(1.0, current_importance * 1.10)
        await self.pg.execute(
            """
            UPDATE episodic_memories
            SET recall_count = recall_count + 1,
                last_recalled = $3,
                importance = $2
            WHERE id = $1
            """,
            memory_id, new_importance, now,
        )

    async def _invalidate_cache(self, user_id: str) -> None:
        """Clear retrieval cache for a user after new memories are stored."""
        pattern = f"episodic_cache:{user_id}:*"
        async for key in self.redis.scan_iter(pattern):
            await self.redis.delete(key)

    async def get_all_unconsolidated(
        self, user_id: str, days_back: int = 30, sim_now: Optional[datetime] = None
    ) -> List[EpisodicMemory]:
        """Fetch all unconsolidated memories for consolidation job."""
        now = sim_now or datetime.now(timezone.utc)
        rows = await self.pg.fetch(
            """
            SELECT * FROM episodic_memories
            WHERE user_id = $1
              AND consolidated = FALSE
              AND created_at > $3 - ($2 * INTERVAL '1 day')
            ORDER BY created_at DESC
            """,
            user_id, days_back, now,
        )
        return [
            EpisodicMemory(
                id=row["id"],
                user_id=user_id,
                content=row["content"],
                memory_type=MemoryType(row["memory_type"]),
                importance=float(row["importance"]),
                recall_count=int(row["recall_count"]),
                tags=list(row["tags"] or []),
                source_turn=row["source_turn"],
                session_id=row["session_id"],
                created_at=row["created_at"],
                last_recalled=row["last_recalled"],
                consolidated=row["consolidated"],
                decay_score=float(row["decay_score"]),
            )
            for row in rows
        ]

    async def mark_consolidated(self, memory_ids: List[UUID]) -> None:
        """Mark episodic memories as consolidated (consumed by semantic tier)."""
        if not memory_ids:
            return
        await self.pg.execute(
            "UPDATE episodic_memories SET consolidated = TRUE WHERE id = ANY($1::uuid[])",
            memory_ids,
        )

    async def delete_all_for_user(self, user_id: str) -> int:
        """GDPR-compliant full delete for a user."""
        result = await self.pg.execute(
            "DELETE FROM episodic_memories WHERE user_id = $1",
            user_id,
        )
        count_str = result.split(" ")[-1]
        return int(count_str) if count_str.isdigit() else 0

    async def update_all_decay_scores(self, user_id: str, sim_now: Optional[datetime] = None) -> str:
        """Recalculate decay scores in SQL using the Ebbinghaus formula."""
        now = sim_now or datetime.now(timezone.utc)
        result = await self.pg.execute(
            """
            UPDATE episodic_memories
            SET decay_score = EXP(
                -EXTRACT(EPOCH FROM ($2 - created_at)) / (86400.0 * 14.0 * (importance + 0.1) * (1 + 0.5 * recall_count))
            )
            WHERE user_id = $1
              AND consolidated = FALSE
              AND expires_at > $2
            """,
            user_id, now,
        )
        return result

    async def delete_expired(self, user_id: str, sim_now: Optional[datetime] = None) -> str:
        """Delete memories past their 30-day TTL."""
        now = sim_now or datetime.now(timezone.utc)
        result = await self.pg.execute(
            "DELETE FROM episodic_memories WHERE user_id = $1 AND expires_at < $2",
            user_id, now,
        )
        return result

    # ── v2 Helpers (Surprise Gating + Contradiction Detection) ───────

    async def get_recent_contents(
        self, user_id: str, limit: int = 50
    ) -> List[str]:
        """
        Get recent memory content strings for surprise scoring.
        Used by SurpriseScorer to compute novelty against existing state.
        """
        rows = await self.pg.fetch(
            """
            SELECT content FROM episodic_memories
            WHERE user_id = $1
              AND decay_score > 0.1
              AND expires_at > NOW()
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
        return [row["content"] for row in rows]

    async def get_recent_contents_with_ids(
        self, user_id: str, limit: int = 50
    ) -> tuple:
        """
        Get recent memory content AND IDs for contradiction detection.

        Returns:
            (contents: List[str], ids: List[str])

        The IDs are needed so the contradiction detector can tell
        the episodic store WHICH specific memory to invalidate.
        Without IDs, contradictions are detected but never resolved.
        """
        rows = await self.pg.fetch(
            """
            SELECT id, content FROM episodic_memories
            WHERE user_id = $1
              AND decay_score > 0.1
              AND expires_at > NOW()
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
        contents = [row["content"] for row in rows]
        ids = [str(row["id"]) for row in rows]
        return contents, ids

    async def invalidate_memory(self, memory_id: str) -> None:
        """
        Invalidate a memory (contradiction detected).
        Sets decay_score to 0 but preserves the record for provenance.
        """
        await self.pg.execute(
            """
            UPDATE episodic_memories
            SET decay_score = 0.0
            WHERE id = $1::uuid
            """,
            memory_id,
        )
        logger.info("memory_invalidated", extra={"memory_id": memory_id})

    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType,
        importance: float,
        source_turn: int,
        session_id: UUID,
        created_at: Optional[datetime] = None,
    ) -> None:
        """
        Direct single-memory store (used by the v2 background pipeline
        after surprise gating and contradiction checks have passed).
        """
        now = created_at or datetime.now(timezone.utc)
        expires_at = now + timedelta(days=30)
        await self.pg.execute(
            """
            INSERT INTO episodic_memories
                (user_id, content, memory_type, importance,
                 recall_count, session_id, source_turn, created_at, expires_at, decay_score)
            VALUES ($1, $2, $3, $4, 0, $5, $6, $7, $8, 1.0)
            """,
            user_id, content, memory_type.value, importance,
            session_id, source_turn, now, expires_at,
        )
        await self._invalidate_cache(user_id)

