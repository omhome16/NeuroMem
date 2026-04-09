"""
Procedural Memory Store — Redis-backed (Tier 5).

Tracks HOW the user likes to interact, not just WHAT they said.
This is the 5th of 6 cognitive memory types identified in modern
AI memory research (Short-term, Long-term, Episodic, Semantic,
Procedural, Working).

Procedural memories capture:
- Communication style preferences ("prefers concise responses")
- Workflow patterns ("always asks for code examples first")
- Behavioral tendencies ("gets frustrated with long explanations")
- Interaction rituals ("likes bullet points over paragraphs")

Storage: Redis hash at `procedural:{user_id}`
- Fast read on every request (~0.1ms)
- Updated periodically (every N turns) to avoid LLM cost
- Small payload (1-5 sentences)

Reference: MACLA (AAMAS 2026) — hierarchical procedural memory
for agents achieves "almost linear mastery" of tasks.
"""
import json
import logging
from typing import Optional, List
from datetime import datetime, timezone

import redis.asyncio as aioredis

from app.config import get_settings
from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings()

# How often to re-extract procedural patterns (every N user turns)
PROCEDURAL_UPDATE_INTERVAL = 10

PROCEDURAL_EXTRACTION_PROMPT = """You are analyzing a user's interaction patterns to build a behavioral profile.

Given the following recent conversation turns, identify the user's:
1. Communication style (concise vs detailed, formal vs casual)
2. Content preferences (code examples, bullet points, step-by-step, etc.)
3. Behavioral patterns (asks follow-ups, prefers autonomy, likes explanations)

Return a JSON object with EXACTLY these fields:
{
    "style": "brief description of their communication style",
    "preferences": ["list", "of", "content", "preferences"],
    "patterns": ["list", "of", "behavioral", "patterns"],
    "summary": "One sentence capturing how this user likes to interact"
}

Be specific and evidence-based. Only include patterns you can clearly observe.
If you cannot determine a pattern, omit it. Return empty lists if unsure."""


class ProceduralMemoryStore:
    """
    Redis-backed store for user interaction patterns and preferences.

    Unlike episodic memory (what happened), procedural memory captures
    how the user prefers to work — enabling the AI to adapt its
    communication style and response format automatically.
    """

    REDIS_KEY_TEMPLATE = "procedural:{user_id}"
    TURN_COUNTER_KEY = "procedural_turns:{user_id}"

    def __init__(self, redis_client: aioredis.Redis, llm_client: LLMClient):
        self.redis = redis_client
        self.llm = llm_client

    async def get_procedural_context(self, user_id: str) -> Optional[str]:
        """
        Retrieve the user's procedural memory for injection into system prompt.
        Returns a formatted string or None if no procedural memory exists.

        This is called on EVERY request — must be fast (<1ms).
        """
        key = self.REDIS_KEY_TEMPLATE.format(user_id=user_id)
        raw = await self.redis.get(key)

        if not raw:
            return None

        try:
            data = json.loads(raw)
            parts = []

            if data.get("summary"):
                parts.append(f"User interaction style: {data['summary']}")

            if data.get("preferences"):
                prefs = ", ".join(data["preferences"][:5])
                parts.append(f"Content preferences: {prefs}")

            if data.get("patterns"):
                patterns = ", ".join(data["patterns"][:3])
                parts.append(f"Behavioral patterns: {patterns}")

            return " | ".join(parts) if parts else None

        except (json.JSONDecodeError, KeyError):
            return None

    async def should_update(self, user_id: str) -> bool:
        """
        Check if we should re-extract procedural patterns.
        Returns True every PROCEDURAL_UPDATE_INTERVAL turns.
        """
        counter_key = self.TURN_COUNTER_KEY.format(user_id=user_id)
        count = await self.redis.incr(counter_key)

        # Set expiry on counter so it doesn't persist forever
        if count == 1:
            await self.redis.expire(counter_key, 86400)  # 24h TTL

        return count % PROCEDURAL_UPDATE_INTERVAL == 0

    async def extract_and_store(
        self,
        user_id: str,
        recent_turns: List[dict],
    ) -> None:
        """
        Extract procedural patterns from recent turns and store in Redis.

        Called from the background pipeline every N turns.
        Uses a single LLM call to analyze interaction patterns.
        """
        if not recent_turns or len(recent_turns) < 4:
            return  # Need enough context to detect patterns

        # Format turns for analysis
        turn_text = "\n".join(
            f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content'][:200]}"
            for t in recent_turns[-20:]  # Last 20 turns max
        )

        try:
            raw = await self.llm.complete(
                user_content=f"CONVERSATION TURNS:\n{turn_text}\n\nAnalyze this user's interaction patterns.",
                system=PROCEDURAL_EXTRACTION_PROMPT,
                temperature=0.0,
            )

            # Parse the response
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            data = json.loads(cleaned)

            # Validate structure
            result = {
                "style": data.get("style", ""),
                "preferences": data.get("preferences", [])[:5],
                "patterns": data.get("patterns", [])[:5],
                "summary": data.get("summary", ""),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "turn_count": len(recent_turns),
            }

            # Store in Redis (no TTL — procedural memory is permanent)
            key = self.REDIS_KEY_TEMPLATE.format(user_id=user_id)
            await self.redis.set(key, json.dumps(result))

            logger.info(
                "procedural_memory_updated",
                extra={
                    "user_id": user_id,
                    "summary": result["summary"][:80],
                    "preferences_count": len(result["preferences"]),
                },
            )

        except json.JSONDecodeError as e:
            logger.warning(
                "procedural_extraction_parse_error",
                extra={"error": str(e)},
            )
        except Exception as e:
            logger.warning(
                "procedural_extraction_failed",
                extra={"error": str(e)},
            )

    async def delete_for_user(self, user_id: str) -> None:
        """GDPR-compliant deletion of procedural memory."""
        key = self.REDIS_KEY_TEMPLATE.format(user_id=user_id)
        counter_key = self.TURN_COUNTER_KEY.format(user_id=user_id)
        await self.redis.delete(key, counter_key)
