"""
Redis async connection pool singleton.

Used for Working Memory (Tier 1) and Episodic retrieval cache.
"""
import logging
from typing import Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def create_redis_pool() -> aioredis.Redis:
    """Create the global Redis connection."""
    global _redis
    _redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=20,
    )
    await _redis.ping()
    logger.info("redis_connected")
    return _redis


async def get_redis_client() -> aioredis.Redis:
    """Get the global Redis client. Creates one if not exists."""
    global _redis
    if _redis is None:
        _redis = await create_redis_pool()
    return _redis


async def close_redis_pool() -> None:
    """Close the global Redis connection."""
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
        logger.info("redis_closed")
