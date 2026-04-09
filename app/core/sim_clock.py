from datetime import datetime, timezone, timedelta
from typing import Optional
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)

class SimulatedClock:
    """
    Manages a simulated "Current Time" for each user.
    Allows testing temporal memory features (decay, time skips) 
    by offsetting the system clock.
    """
    
    CLOCK_KEY_PREFIX = "clock_offset:"

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def get_offset_days(self, user_id: str) -> int:
        """Fetch the current clock offset in days for this user."""
        key = f"{self.CLOCK_KEY_PREFIX}{user_id}"
        offset = await self.redis.get(key)
        return int(offset) if offset else 0

    async def increment_offset(self, user_id: str, days: int) -> int:
        """Add days to the current offset."""
        key = f"{self.CLOCK_KEY_PREFIX}{user_id}"
        return await self.redis.incrby(key, days)

    async def get_now(self, user_id: str) -> datetime:
        """Get the simulated "Now" for this user."""
        offset_days = await self.get_offset_days(user_id)
        now = datetime.now(timezone.utc)
        if offset_days:
            now += timedelta(days=offset_days)
        return now

    async def get_now_iso(self, user_id: str) -> str:
        """Get simulated "Now" as an ISO string (for Neo4j/Postgres directly)."""
        now = await self.get_now(user_id)
        return now.isoformat()

    async def get_now_pretty(self, user_id: str) -> str:
        """Get simulated "Now" in a human-readable format for system prompts."""
        now = await self.get_now(user_id)
        return now.strftime("%A, %B %d, %Y at %I:%M %p UTC")
