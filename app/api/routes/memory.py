"""
Memory CRUD endpoints.

- POST /memory/ingest: Manually ingest a memory
- GET /memory/retrieve: Search memories by query
- GET /memory/timeline: View episodic memory timeline
- DELETE /memory/clear: GDPR-compliant full wipe
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.middleware.auth import verify_api_key
from app.dependencies import get_episodic_store, get_memory_router, get_semantic_store
from app.memory.episodic.store import EpisodicMemoryStore
from app.memory.semantic.store import SemanticMemoryStore
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
    graph: __import__('app.memory.graph.knowledge_graph', fromlist=['KnowledgeGraph']).KnowledgeGraph = Depends(__import__('app.dependencies', fromlist=['get_knowledge_graph']).get_knowledge_graph),
):
    """Dump the entire state of all 4 memory tiers for the user."""
    from app.db.redis_client import get_redis_client
    
    # 1. Working Memory (Redis)
    redis = await get_redis_client()
    working_data = []
    async for key in redis.scan_iter(f"working:{user_id}:*"):
        session_data = await redis.get(key)
        if session_data:
            import json
            try:
                working_data.append(json.loads(session_data))
            except:
                working_data.append(session_data.decode("utf-8") if isinstance(session_data, bytes) else session_data)
            
    # 2. Episodic Memory (Postgres)
    episodes = await episodic_store.get_all_unconsolidated(user_id=user_id, days_back=365)
    episodic_data = [{"content": e.content, "importance": e.importance, "decay": e.decay_score, "created": str(e.created_at)} for e in episodes]
    
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
        "graph": graph_data.get("edges", [])
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
):
    """
    GDPR-compliant full memory wipe.
    Deletes: working memory (Redis), episodic (PostgreSQL), semantic (Qdrant).
    """
    # Clear episodic (PostgreSQL)
    deleted_episodic = await episodic_store.delete_all_for_user(user_id)

    # Clear semantic (Qdrant)
    await semantic_store.delete_user_memories(user_id)

    # Clear working memory sessions (Redis pattern delete)
    redis = await get_redis_client()
    async for key in redis.scan_iter("working:*"):
        await redis.delete(key)

    logger.info(
        "gdpr_memory_cleared",
        extra={"user_id": user_id, "episodic_deleted": deleted_episodic},
    )

    return {
        "status": "cleared",
        "user_id": user_id,
        "episodic_deleted": deleted_episodic,
    }
