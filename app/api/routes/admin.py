"""
Admin endpoints — replaces Celery workers with manual triggers.

Uses LangChain for the consolidation LLM call with structured output.

- POST /admin/consolidate: Promote episodic memories to semantic tier
- POST /admin/decay-update: Recalculate Ebbinghaus decay scores
- POST /admin/cleanup: Delete expired memories
"""
import json
import logging
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.middleware.auth import verify_api_key
from app.config import get_settings
from app.core.llm_client import LLMClient
from app.dependencies import (
    get_episodic_store,
    get_semantic_store,
    get_llm_client,
)
from app.memory.episodic.scorer import EbbinghausScorer
from app.memory.episodic.store import EpisodicMemoryStore
from app.memory.semantic.store import SemanticMemoryStore
from app.models.memory import EpisodicMemory, MemoryType, SemanticMemory
from app.memory.graph.knowledge_graph import KnowledgeGraph
from app.core.consolidation_graph import ConsolidationGraph
from app.dependencies import get_entity_extractor, get_contradiction_detector, get_surprise_scorer, get_knowledge_graph

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Pydantic Schema for LangChain Structured Output ──────────────────

class SemanticFact(BaseModel):
    """A single consolidated semantic fact about the user."""
    content: str = Field(description="A clear, standalone factual sentence about the user")
    importance: float = Field(ge=0.0, le=1.0, description="Importance score")
    type: str = Field(description="Memory type: fact, preference, event, or relationship")


class ConsolidationResponse(BaseModel):
    """Structured response from the LangChain consolidation chain."""
    semantic_facts: List[SemanticFact] = Field(
        default_factory=list,
        description="Core semantic facts distilled from episodic memories"
    )


CONSOLIDATION_PROMPT = """You are a memory consolidation engine.
Given a list of episodic memory entries (facts observed across many conversations),
your job is to distill them into 5-10 core, high-signal semantic facts about the user.

Rules:
- Merge redundant facts (e.g., multiple mentions of "likes coffee" → one fact)
- Resolve any remaining contradictions (pick more recent or more important)
- Omit trivial or low-importance memories
- Write each fact as a clear, standalone sentence"""


@router.post("/consolidate")
async def consolidate(
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
    semantic_store: SemanticMemoryStore = Depends(get_semantic_store),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """
    Manually trigger memory consolidation: episodic → semantic.

    Uses LangGraph multi-stage pipeline:
    1. Score memories (Ebbinghaus decay filtering)
    2. Cluster by topic
    3. Extract graph updates (entity relations)
    4. Detect contradictions
    5. Compress to semantic (LLM summarization)
    6. Compute surprise deltas
    """
    start_time = datetime.now(timezone.utc)
    
    # Step 1: Fetch unconsolidated memories
    episodes = await episodic_store.get_all_unconsolidated(user_id=user_id)

    if not episodes:
        return {"status": "no_memories", "message": "No unconsolidated memories found."}

    from app.dependencies import get_entity_extractor, get_contradiction_detector, get_surprise_scorer
    
    consolidation_graph = ConsolidationGraph(
        llm_client=llm_client,
        entity_extractor=await get_entity_extractor(),
        contradiction_detector=get_contradiction_detector(),
        surprise_scorer=get_surprise_scorer()
    )

    result = await consolidation_graph.run(user_id, episodes)
    
    keep = result["memories_to_keep"]
    delete_candidates = result["memories_to_delete"]
    semantic_facts = result["semantic_facts"]
    triples = result["graph_triples"]

    # Delete low-decay memories
    deleted_count = 0
    if delete_candidates:
        from app.db.postgres import get_pg_pool
        pg = await get_pg_pool()
        delete_ids = [m.id for m in delete_candidates]
        await pg.execute(
            "DELETE FROM episodic_memories WHERE id = ANY($1::uuid[])",
            delete_ids,
        )
        deleted_count = len(delete_ids)

    # Write to Qdrant
    facts_written = 0
    from app.db.redis_client import get_redis_client
    from app.core.sim_clock import SimulatedClock
    redis = await get_redis_client()
    clock = SimulatedClock(redis)
    sim_now = await clock.get_now(user_id)
    sim_now_iso = await clock.get_now_iso(user_id)

    if semantic_facts:
        semantic_memories = [
            SemanticMemory(
                id=str(uuid4()),
                user_id=user_id,
                content=fact["content"],
                memory_type=MemoryType(fact.get("type", "fact")),
                importance=float(fact.get("importance", 0.7)),
                created_at=sim_now,
                source_episode_ids=fact.get("source_ids", []),
            )
            for fact in semantic_facts
        ]
        facts_written = await semantic_store.upsert_memories(user_id, semantic_memories)

    if triples:
        await knowledge_graph.add_triples(user_id, triples, now_iso=sim_now_iso)

    if keep:
        await episodic_store.mark_consolidated([m.id for m in keep])

    duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

    # Log the run
    from app.db.postgres import get_pg_pool
    pg = await get_pg_pool()
    await pg.execute(
        """
        INSERT INTO consolidation_runs
            (user_id, episodes_processed, facts_written, memories_deleted, duration_ms)
        VALUES ($1, $2, $3, $4, $5)
        """,
        user_id, len(episodes), facts_written, deleted_count, duration_ms,
    )

    return {
        "status": "success",
        "episodes_processed": len(episodes),
        "memories_kept": len(keep),
        "memories_deleted": deleted_count,
        "semantic_facts_written": facts_written,
        "graph_triples_added": len(triples),
        "duration_ms": duration_ms,
    }


@router.post("/time-skip")
async def time_skip(
    days: int = 15,
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
):
    """Artificially age all episodic memories by X days to test Ebbinghaus decay natively."""
    from app.db.postgres import get_pg_pool
    from app.db.redis_client import get_redis_client
    from app.core.sim_clock import SimulatedClock
    
    # 1. Update existing memory timestamps in Postgres
    pg = await get_pg_pool()
    await pg.execute(
        "UPDATE episodic_memories SET created_at = created_at - ($2 * INTERVAL '1 day') WHERE user_id = $1",
        user_id, days
    )
    
    # 2. Update the Simulated Clock offset in Redis (so new facts are timestamped in the "future")
    redis = await get_redis_client()
    clock = SimulatedClock(redis)
    new_offset = await clock.increment_offset(user_id, days)
    
    # 3. Recalculate decay scores based on new time
    result = await episodic_store.update_all_decay_scores(user_id, sim_now=sim_now)
    
    return {
        "status": "success", 
        "days_skipped": days, 
        "total_offset_days": new_offset,
        "decay_recalculated": result
    }


@router.post("/decay-update")
async def update_decay_scores(
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
):
    """Recalculate Ebbinghaus decay scores for all episodic memories."""
    from app.db.redis_client import get_redis_client
    from app.core.sim_clock import SimulatedClock
    redis = await get_redis_client()
    clock = SimulatedClock(redis)
    sim_now = await clock.get_now(user_id)
    
    result = await episodic_store.update_all_decay_scores(user_id, sim_now=sim_now)
    return {"status": "updated", "result": result}


@router.post("/cleanup")
async def cleanup_expired(
    user_id: str = Depends(verify_api_key),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
):
    """Delete expired episodic memories (past 30-day TTL)."""
    from app.db.redis_client import get_redis_client
    from app.core.sim_clock import SimulatedClock
    redis = await get_redis_client()
    clock = SimulatedClock(redis)
    sim_now = await clock.get_now(user_id)
    
    result = await episodic_store.delete_expired(user_id, sim_now=sim_now)
    return {"status": "cleaned", "result": result}


async def _llm_consolidate(
    episodes: List[EpisodicMemory], llm_client: LLMClient
) -> List[dict]:
    """
    Use LangChain structured output to compress episodes into semantic facts.

    Attempts with_structured_output() first, falls back to raw JSON parsing.
    """
    memory_list = "\n".join(
        f"- [{m.memory_type.value}] (importance={m.importance:.1f}, "
        f"recalled={m.recall_count}x) {m.content}"
        for m in episodes
    )

    prompt = f"""USER'S EPISODIC MEMORIES ({len(episodes)} entries):
{memory_list}

Consolidate these into core semantic facts about this user."""

    # Attempt 1: LangChain structured output via with_structured_output()
    result = await llm_client.structured_output(
        user_content=prompt,
        output_schema=ConsolidationResponse,
        system=CONSOLIDATION_PROMPT,
        temperature=0.0,
    )

    if result and isinstance(result, ConsolidationResponse):
        logger.info("consolidation_structured_success", extra={"facts": len(result.semantic_facts)})
        return [
            {"content": f.content, "importance": f.importance, "type": f.type}
            for f in result.semantic_facts
        ]

    # Fallback: raw LangChain completion + JSON parsing
    logger.info("consolidation_structured_fallback")
    raw = await llm_client.complete(
        prompt,
        system=CONSOLIDATION_PROMPT,
        temperature=0.0,
    )

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        data = json.loads(cleaned)
        return data.get("semantic_facts", [])
    except json.JSONDecodeError as e:
        logger.error("consolidation_parse_error", extra={"error": str(e)})
        return []
