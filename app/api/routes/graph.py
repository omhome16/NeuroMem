"""
Graph API endpoints — Neo4j knowledge graph querying.

- GET /graph: Retrieve the user's full knowledge graph
- GET /graph/query: Query graph for specific entities
- DELETE /graph: Delete user's graph (GDPR)
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from app.api.middleware.auth import verify_api_key
from app.dependencies import get_knowledge_graph
from app.memory.graph.knowledge_graph import KnowledgeGraph

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("")
async def get_user_graph(
    include_invalidated: bool = Query(False, description="Include temporally invalidated facts"),
    user_id: str = Depends(verify_api_key),
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
):
    """
    Retrieve the user's full temporal knowledge graph.

    Returns nodes and edges formatted for visualization.
    Active facts have is_active=true, invalidated facts have is_active=false.
    """
    data = await graph.get_user_graph(
        user_id=user_id,
        include_invalidated=include_invalidated,
    )
    return data


@router.get("/query")
async def query_graph(
    entities: str = Query(..., description="Comma-separated entity names to query"),
    max_hops: int = Query(2, ge=1, le=4),
    user_id: str = Depends(verify_api_key),
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
):
    """Query the knowledge graph for facts about specific entities."""
    entity_list = [e.strip() for e in entities.split(",") if e.strip()]
    facts = await graph.query_context(
        user_id=user_id,
        query_entities=entity_list,
        max_hops=max_hops,
    )
    return {"entities": entity_list, "facts": facts, "count": len(facts)}


@router.delete("")
async def delete_user_graph(
    user_id: str = Depends(verify_api_key),
    graph: KnowledgeGraph = Depends(get_knowledge_graph),
):
    """GDPR: Delete the user's entire knowledge graph."""
    deleted = await graph.delete_user_graph(user_id)
    return {"status": "deleted", "nodes_deleted": deleted}
