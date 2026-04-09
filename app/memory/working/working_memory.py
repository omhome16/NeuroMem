"""
Working Memory — Redis-backed sliding window (Tier 1).

Maintains the last N conversation turns for a session.
When turn count exceeds threshold, older turns are compressed
into a summary by the LLM.

Keys:
    working:{session_id}:turns   → JSON list of turns (TTL: 2h)
    working:{session_id}:summary → compressed summary string (TTL: 2h)
    working:{session_id}:meta    → {user_id, turn_count} (TTL: 2h)
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID, uuid4

import redis.asyncio as aioredis

from app.config import get_settings
from app.models.memory import ConversationTurn, WorkingMemoryState
from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings()


class WorkingMemory:
    """
    Redis-backed sliding window conversation memory.
    Provides short-term context for the current session.
    """

    TURNS_KEY = "working:{session_id}:turns"
    SUMMARY_KEY = "working:{session_id}:summary"
    META_KEY = "working:{session_id}:meta"
    COMPRESS_AT_TURN = 10  # Start compression at turn 10

    def __init__(self, redis_client: aioredis.Redis, llm_client: LLMClient):
        self.redis = redis_client
        self.llm = llm_client
        self.ttl = settings.working_memory_ttl_seconds
        self.max_turns = settings.working_memory_max_turns

    def _turns_key(self, session_id: UUID) -> str:
        return self.TURNS_KEY.format(session_id=str(session_id))

    def _summary_key(self, session_id: UUID) -> str:
        return self.SUMMARY_KEY.format(session_id=str(session_id))

    def _meta_key(self, session_id: UUID) -> str:
        return self.META_KEY.format(session_id=str(session_id))

    async def get_or_create_session(
        self, user_id: str, session_id: Optional[UUID] = None
    ) -> UUID:
        """Return existing session ID or create a new one."""
        if session_id is None:
            session_id = uuid4()

        meta_key = self._meta_key(session_id)
        existing = await self.redis.get(meta_key)
        if not existing:
            await self.redis.setex(
                meta_key,
                self.ttl,
                json.dumps({"user_id": user_id, "turn_count": 0}),
            )
        return session_id

    async def add_turn(
        self, session_id: UUID, role: str, content: str
    ) -> ConversationTurn:
        """Append a turn to working memory. Triggers compression if needed."""
        turns_key = self._turns_key(session_id)
        meta_key = self._meta_key(session_id)

        # Deserialize existing turns
        raw = await self.redis.get(turns_key)
        turns: List[dict] = json.loads(raw) if raw else []

        meta_raw = await self.redis.get(meta_key)
        meta = json.loads(meta_raw) if meta_raw else {"turn_count": 0}

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            turn_index=meta["turn_count"],
        )
        turns.append(turn.model_dump(mode="json"))

        # Sliding window — keep only max_turns
        if len(turns) > self.max_turns:
            turns = turns[-self.max_turns:]

        meta["turn_count"] += 1

        # Trigger semantic compression at threshold
        if meta["turn_count"] == self.COMPRESS_AT_TURN:
            await self._compress_to_summary(session_id, turns)

        # Persist with TTL
        pipe = self.redis.pipeline()
        pipe.setex(turns_key, self.ttl, json.dumps(turns))
        pipe.setex(meta_key, self.ttl, json.dumps(meta))
        await pipe.execute()

        return turn

    async def get_turns(self, session_id: UUID) -> List[ConversationTurn]:
        """Retrieve current turns for a session."""
        raw = await self.redis.get(self._turns_key(session_id))
        if not raw:
            return []
        return [ConversationTurn(**t) for t in json.loads(raw)]

    async def get_summary(self, session_id: UUID) -> Optional[str]:
        """Return compressed summary if it exists."""
        raw = await self.redis.get(self._summary_key(session_id))
        return raw if raw else None

    async def _compress_to_summary(
        self, session_id: UUID, turns: List[dict]
    ) -> None:
        """LLM-compress older turns into a brief summary."""
        conversation_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns[:-4]
        )
        prompt = f"""Summarize the following conversation in 3-5 sentences.
Focus on key facts, decisions, and preferences the user expressed.
Be concise and factual.

CONVERSATION:
{conversation_text}

SUMMARY:"""

        try:
            summary = await self.llm.complete(prompt)
            await self.redis.setex(
                self._summary_key(session_id),
                self.ttl,
                summary.strip(),
            )
            logger.info("working_memory_compressed", extra={"session_id": str(session_id)})
        except Exception as e:
            logger.error("compression_failed", extra={"error": str(e)})

    async def get_context_for_prompt(
        self, session_id: UUID
    ) -> WorkingMemoryState:
        """Return the full working memory state for prompt injection."""
        turns = await self.get_turns(session_id)
        summary = await self.get_summary(session_id)
        return WorkingMemoryState(
            session_id=session_id,
            user_id="default",  # filled by caller
            turns=turns,
            compressed_summary=summary,
        )

    async def clear_session(self, session_id: UUID) -> None:
        """Delete all working memory for a session."""
        pipe = self.redis.pipeline()
        pipe.delete(self._turns_key(session_id))
        pipe.delete(self._summary_key(session_id))
        pipe.delete(self._meta_key(session_id))
        await pipe.execute()
        logger.info("session_cleared", extra={"session_id": str(session_id)})
