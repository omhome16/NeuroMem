"""
Neo4j async driver singleton.

Used for the Temporal Knowledge Graph — stores entity-relationship
triples with temporal validity (valid_from/valid_to).
"""
import logging
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_driver: Optional[AsyncDriver] = None


async def init_neo4j() -> AsyncDriver:
    """Initialize the global Neo4j async driver."""
    global _driver
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    # Verify connectivity
    async with _driver.session() as session:
        await session.run("RETURN 1")
    logger.info("neo4j_connected", extra={"uri": settings.neo4j_uri})

    # Create constraints and indexes
    async with _driver.session() as session:
        await session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        await session.run(
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.user_id)"
        )
    return _driver


def get_neo4j_driver() -> AsyncDriver:
    """Get the global Neo4j driver."""
    global _driver
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialized. Call init_neo4j() first.")
    return _driver


async def close_neo4j() -> None:
    """Close the global Neo4j driver."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
        logger.info("neo4j_closed")
