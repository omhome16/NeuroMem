"""
Memory CRUD endpoints.

- POST /memory/ingest: Manually ingest a memory
- GET /memory/retrieve: Search memories by query
- GET /memory/timeline: View episodic memory timeline
- DELETE /memory/clear: GDPR-compliant full wipe
"""
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.dependencies import (
    get_episodic_store,
    get_memory_router,
    get_semantic_store,
    get_knowledge_graph,
)
from app.memory.episodic.store import EpisodicMemoryStore
from app.memory.semantic.store import SemanticMemoryStore
from app.memory.graph.knowledge_graph import KnowledgeGraph
from app.core.memory_router import MemoryRouter
from app.models.memory import RetrievedMemory, MemoryType, ExtractedMemory
from app.db.redis_client import get_redis_client

router = APIRouter()
logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    content: str
    memory_type: str = "fact"
    importance: float = 0.5


@router.post("/ingest")
async def ingest_memory(
    request: IngestRequest,
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
):
    """Manually ingest a memory for a user."""
    from uuid import uuid4

    extracted = ExtractedMemory(
        content=request.content,
        importance=request.importance,
        memory_type=MemoryType(request.memory_type),
    )

    stored = await episodic_store.store_memories(
        user_id=user_id,
        extracted=[extracted],
        session_id=uuid4(),
        turn_index=0,
    )

    return {
        "status": "ingested",
        "user_id": user_id,
        "memories_stored": len(stored),
    }


@router.get("/retrieve", response_model=List[RetrievedMemory])
async def retrieve_memories(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, le=50),
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
    semantic_store: SemanticMemoryStore = Depends(get_semantic_store),
):
    """Retrieve relevant memories from episodic and semantic tiers."""
    # Get from both tiers
    episodic_results = await episodic_store.retrieve(
        user_id=user_id,
        query_text=query,
        limit=limit,
    )

    semantic_results = await semantic_store.search(
        user_id=user_id,
        query_text=query,
        limit=limit,
    )

    # Merge and sort by relevance
    all_results = episodic_results + semantic_results
    all_results.sort(key=lambda m: m.relevance_score, reverse=True)
    return all_results[:limit]


@router.get("/state")
async def get_memory_state(
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
    semantic_store: SemanticMemoryStore = Depends(get_semantic_store),
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
):
    """Dump the entire state of all 4 memory tiers for the user."""
    # 1. Working Memory (Redis)
    redis = await get_redis_client()
    working_data = []
    async for key in redis.scan_iter(f"working:{user_id}:*:turns"):
        session_data = await redis.get(key)
        if session_data:
            try:
                turns = json.loads(session_data)
                working_data.extend(turns)
            except (json.JSONDecodeError, TypeError):
                pass

    # 2. Episodic Memory (Postgres)
    episodes = await episodic_store.get_all_unconsolidated(user_id=user_id, days_back=365)
    episodic_data = [
        {
            "content": e.content,
            "importance": e.importance,
            "decay": e.decay_score,
            "type": e.memory_type.value,
            "recall_count": e.recall_count,
            "created": str(e.created_at),
        }
        for e in episodes
    ]

    # 3. Semantic Memory (Qdrant)
    semantic_data = []
    if hasattr(semantic_store, 'get_all_for_user'):
        semantics = await semantic_store.get_all_for_user(user_id=user_id)
        semantic_data = [{"content": s.content, "importance": s.importance} for s in semantics]

    # 4. Graph Memory (Neo4j)
    graph_data = await graph.get_user_graph(user_id=user_id)

    return {
        "working": working_data,
        "episodic": episodic_data,
        "semantic": semantic_data,
        "graph": graph_data.get("edges", []),
        "graph_nodes": graph_data.get("nodes", []),
        "graph_node_count": graph_data.get("node_count", 0),
        "graph_edge_count": graph_data.get("edge_count", 0),
    }


@router.get("/timeline")
async def get_memory_timeline(
    days_back: int = Query(30, le=90),
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
):
    """Return episodic memory timeline for the user."""
    memories = await episodic_store.get_all_unconsolidated(
        user_id=user_id,
        days_back=days_back,
    )
    return [
        {
            "id": str(m.id),
            "content": m.content,
            "type": m.memory_type.value,
            "importance": m.importance,
            "decay_score": m.decay_score,
            "recall_count": m.recall_count,
            "created_at": m.created_at.isoformat(),
            "consolidated": m.consolidated,
        }
        for m in memories
    ]


@router.delete("/clear")
async def clear_memories(
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
    semantic_store: SemanticMemoryStore = Depends(get_semantic_store),
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
):
    """
    GDPR-compliant full memory wipe.
    Deletes: working memory (Redis), episodic (PostgreSQL), semantic (Qdrant).
    """
    # Clear episodic (PostgreSQL)
    deleted_episodic = await episodic_store.delete_all_for_user(user_id)

    # Clear semantic (Qdrant)
    await semantic_store.delete_user_memories(user_id)

    # Clear knowledge graph (Neo4j)
    graph_deleted = 0
    try:
        graph_deleted = await graph.delete_user_graph(user_id)
    except Exception as e:
        logger.warning("graph_clear_failed", extra={"error": str(e)})

    # Clear working memory sessions (Redis)
    # Also clear clock offsets and procedural memory for this user
    redis = await get_redis_client()
    keys_deleted = 0
    for pattern in [
        f"working:{user_id}:*:turns", f"working:{user_id}:*:summary", f"working:{user_id}:*:meta",
        f"procedural:{user_id}", f"procedural_turns:{user_id}",
        f"clock_offset:{user_id}", f"episodic_cache:{user_id}:*",
    ]:
        async for key in redis.scan_iter(pattern):
            await redis.delete(key)
            keys_deleted += 1

    logger.info(
        "gdpr_memory_cleared",
        extra={
            "user_id": user_id,
            "episodic_deleted": deleted_episodic,
            "redis_keys_deleted": keys_deleted,
        },
    )

    return {
        "status": "cleared",
        "user_id": user_id,
        "episodic_deleted": deleted_episodic,
        "redis_keys_deleted": keys_deleted,
        "graph_nodes_deleted": graph_deleted,
    }


@router.get("/auth-diag")
async def auth_diag(user_id: str = Depends(verify_api_key)):
    return {"user_id": user_id}
