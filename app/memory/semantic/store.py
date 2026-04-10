"""
Semantic Memory Store — Qdrant-backed (Tier 3).

Long-term permanent vector store for consolidated user knowledge.
Uses cosine similarity search with user_id payload filtering.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings
from app.memory.semantic.embedder import Embedder
from app.models.memory import MemoryType, RetrievedMemory, SemanticMemory, MemoryTier

logger = logging.getLogger(__name__)
settings = get_settings()


class SemanticMemoryStore:
    """
    Qdrant-backed long-term semantic memory store.

    Collection schema:
        vector: N-dim embeddings (cosine similarity)
        payload: {user_id, content, memory_type, importance, created_at, source_episode_ids}
    """

    def __init__(self, client: AsyncQdrantClient, embedder: Embedder):
        self.client = client
        self.embedder = embedder
        self.collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload index for user_id filtering
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="user_id",
                field_schema="keyword",
            )
            logger.info("qdrant_collection_created", extra={"collection": self.collection})

    async def upsert_memories(
        self,
        user_id: str,
        memories: List[SemanticMemory],
    ) -> int:
        """
        Upsert semantic memories into Qdrant.
        Returns count of successfully upserted memories.
        """
        if not memories:
            return 0

        texts = [m.content for m in memories]
        embeddings = self.embedder.embed_batch(texts)

        points = []
        for memory, embedding in zip(memories, embeddings):
            points.append(
                PointStruct(
                    id=str(memory.id),
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type.value,
                        "importance": memory.importance,
                        "created_at": memory.created_at.isoformat(),
                        "source_episode_ids": memory.source_episode_ids,
                    },
                )
            )

        await self.client.upsert(
            collection_name=self.collection,
            points=points,
        )
        logger.info(
            "semantic_memories_upserted",
            extra={"count": len(points), "user_id": user_id},
        )
        return len(points)

    async def search(
        self,
        user_id: str,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[RetrievedMemory]:
        """
        Vector similarity search filtered by user_id.

        Args:
            user_id: User namespace filter
            query_text: Query to embed and search
            limit: Max results
            score_threshold: Minimum cosine similarity
            memory_types: Optional type filter

        Returns:
            List of RetrievedMemory sorted by relevance
        """
        query_embedding = self.embedder.embed(query_text)

        # Build filter: always filter by user_id
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        if memory_types:
            # Filter by first memory type (simplified)
            must_conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=memory_types[0].value),
                )
            )

        results = await self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            query_filter=Filter(must=must_conditions),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        retrieved = []
        for hit in results.points:
            payload = hit.payload
            retrieved.append(
                RetrievedMemory(
                    content=payload["content"],
                    memory_type=MemoryType(payload["memory_type"]),
                    tier=MemoryTier.SEMANTIC,
                    relevance_score=hit.score,
                    importance=payload.get("importance", 0.5),
                    created_at=datetime.fromisoformat(payload["created_at"]),
                )
            )

        return retrieved

    async def get_all_for_user(self, user_id: str) -> List[RetrievedMemory]:
        """Fetch all semantic memories for a given user."""
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        
        # Scroll through all points for the user
        results, next_offset = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=must_conditions),
            limit=1000,
            with_payload=True,
        )
        
        retrieved = []
        for hit in results:
            payload = hit.payload
            retrieved.append(
                RetrievedMemory(
                    content=payload["content"],
                    memory_type=MemoryType(payload["memory_type"]),
                    tier=MemoryTier.SEMANTIC,
                    relevance_score=1.0,
                    importance=payload.get("importance", 0.5),
                    created_at=datetime.fromisoformat(payload["created_at"]),
                )
            )
        return retrieved

    async def delete_user_memories(self, user_id: str) -> None:
        """GDPR: Delete all semantic memories for a user."""
        await self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id", match=MatchValue(value=user_id)
                        )
                    ]
                )
            ),
        )
        logger.info("semantic_memories_deleted", extra={"user_id": user_id})
