"""
Qdrant async client singleton.

Used for Semantic Memory (Tier 3) — permanent vector store.
"""
import logging
from typing import Optional

from qdrant_client import AsyncQdrantClient

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client: Optional[AsyncQdrantClient] = None


async def init_qdrant() -> AsyncQdrantClient:
    """Initialize the global Qdrant client."""
    global _client
    _client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
    )
    logger.info("qdrant_connected", extra={"url": settings.qdrant_url})

    # Ensure collection exists upon connection
    from qdrant_client.http.models import Distance, VectorParams
    collections = await _client.get_collections()
    if settings.qdrant_collection not in [c.name for c in collections.collections]:
        await _client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        await _client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="user_id",
            field_schema="keyword",
        )
        logger.info("qdrant_collection_created", extra={"collection": settings.qdrant_collection})

    return _client


def get_qdrant_client() -> AsyncQdrantClient:
    """Get the global Qdrant client."""
    global _client
    if _client is None:
        raise RuntimeError("Qdrant client not initialized. Call init_qdrant() first.")
    return _client


async def close_qdrant() -> None:
    """Close the global Qdrant client."""
    global _client
    if _client:
        await _client.close()
        _client = None
        logger.info("qdrant_closed")
