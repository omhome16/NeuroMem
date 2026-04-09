"""
PostgreSQL async connection pool singleton.

Uses asyncpg for high-performance async access.
Pool is created once at startup via lifespan() and reused globally.
"""
import logging
from typing import Optional

import asyncpg

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_pool: Optional[asyncpg.Pool] = None


async def create_pg_pool() -> asyncpg.Pool:
    """Create the global PostgreSQL connection pool."""
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("postgres_pool_created")
    return _pool


async def get_pg_pool() -> asyncpg.Pool:
    """Get the global PostgreSQL pool. Creates one if not exists."""
    global _pool
    if _pool is None:
        _pool = await create_pg_pool()
    return _pool


async def close_pg_pool() -> None:
    """Close the global PostgreSQL pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("postgres_pool_closed")
