"""
Chat endpoint — the main interaction point with NeuroMem.

Full v2 pipeline:
1. Auth → validate API key
2. WorkingMemory → get/create session
3. LangGraph MemoryRouter v2 → route + retrieve + rerank + token budget
4. MemoryInjector → build system prompt with tiered memories
5. LangChain ChatOpenAI → generate response
6. WorkingMemory → store turns
7. Background tasks:
   a. MemoryExtractor → extract facts (LangChain structured output)
   b. SurpriseScorer → novelty gate (Titans-inspired)
   c. EntityExtractor → extract graph triples
   d. ContradictionDetector → detect and invalidate contradictions
   e. KnowledgeGraph → update Neo4j
"""
import logging
import time
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.middleware.auth import verify_api_key
from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.memory_injector import MemoryInjector
from app.core.memory_router import MemoryRouter
from app.core.surprise_scorer import SurpriseScorer
from app.core.contradiction_detector import ContradictionDetector
from app.dependencies import (
    get_llm_client,
    get_memory_router,
    get_working_memory,
    get_episodic_store,
    get_knowledge_graph,
    get_entity_extractor,
    get_surprise_scorer,
    get_contradiction_detector,
    get_procedural_store,
)
from app.memory.episodic.extractor import MemoryExtractor
from app.memory.episodic.store import EpisodicMemoryStore
from app.memory.graph.entity_extractor import EntityExtractor
from app.memory.graph.knowledge_graph import KnowledgeGraph
from app.memory.procedural.store import ProceduralMemoryStore
from app.memory.working.working_memory import WorkingMemory
from app.models.chat import ChatRequest, ChatResponse, MemoryDebugInfo
from app.models.memory import MemoryTier

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_api_key),
    llm_client: LLMClient = Depends(get_llm_client),
    working_memory: WorkingMemory = Depends(get_working_memory),
    memory_router: MemoryRouter = Depends(get_memory_router),
    episodic_store: EpisodicMemoryStore = Depends(get_episodic_store),
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph),
    entity_extractor: EntityExtractor = Depends(get_entity_extractor),
    surprise_scorer: SurpriseScorer = Depends(get_surprise_scorer),
    contradiction_detector: ContradictionDetector = Depends(get_contradiction_detector),
    procedural_store: ProceduralMemoryStore = Depends(get_procedural_store),
):
    """
    Send a message with full memory-augmented response.
    """
    from app.core.sim_clock import SimulatedClock
    from app.db.redis_client import get_redis_client
    
    start_time = time.time()
    
    # Init simulated clock
    redis = await get_redis_client()
    clock = SimulatedClock(redis)
    sim_now = await clock.get_now(user_id)
    sim_now_iso = await clock.get_now_iso(user_id)
    sim_now_pretty = await clock.get_now_pretty(user_id)

    # Step 1: Get or create session
    session_id = await working_memory.get_or_create_session(
        user_id=user_id, session_id=request.session_id
    )

    # Step 2: Store user turn in working memory
    await working_memory.add_turn(user_id, session_id, "user", request.message)

    # Step 3: Route and retrieve memories (LangGraph v2 pipeline)
    memories = await memory_router.route_and_retrieve(
        user_id=user_id,
        session_id=session_id,
        query=request.message,
        top_k_per_tier=5,
        sim_now=sim_now,
    )

    # Step 4: Fetch procedural memory (sub-ms Redis read)
    procedural_context = await procedural_store.get_procedural_context(user_id)

    # Step 5: Build system prompt with injected memories + procedural context
    injector = MemoryInjector()
    system_prompt = injector.build_system_prompt(
        memories, 
        procedural_context=procedural_context,
        current_time=sim_now_pretty
    )

    # Step 6: Get conversation history for context
    turns = await working_memory.get_turns(user_id, session_id)
    messages = [{"role": t.role, "content": t.content} for t in turns]

    # Step 6: Call LLM via LangChain
    reply = await llm_client.chat(messages=messages, system=system_prompt)

    # Step 7: Store assistant turn
    await working_memory.add_turn(user_id, session_id, "assistant", reply)

    latency_ms = int((time.time() - start_time) * 1000)

    # Step 9: Background tasks — surprise-gated extraction + graph update + procedural
    background_tasks.add_task(
        _background_memory_pipeline,
        user_id=user_id,
        session_id=session_id,
        user_message=request.message,
        assistant_response=reply,
        turn_index=len(turns),
        llm_client=llm_client,
        episodic_store=episodic_store,
        knowledge_graph=knowledge_graph,
        entity_extractor=entity_extractor,
        surprise_scorer=surprise_scorer,
        contradiction_detector=contradiction_detector,
        procedural_store=procedural_store,
        recent_turns=[{"role": t.role, "content": t.content} for t in turns],
        sim_now=sim_now,
        sim_now_iso=sim_now_iso
    )

    # Step 10: Build enhanced debug info for transparency
    debug_info = None
    if request.include_memory_debug:
        # Step 10: Build enhanced debug info for transparency
        debug_memories = []
        graph_facts = []
        for m in memories:
            debug_memories.append({
                "content": m.content,
                "tier": m.tier.value.upper(),  # Send UPPERCASE for frontend matching
                "type": m.memory_type.value,
                "relevance": round(m.relevance_score, 3),
            })
            if m.tier == MemoryTier.GRAPH and m.content.startswith("[Knowledge Graph]"):
                graph_facts.append(m.content.replace("[Knowledge Graph] ", ""))

        debug_info = MemoryDebugInfo(
            memories_retrieved=len(memories),
            tiers_queried=list(set(m.tier.value.upper() for m in memories)),
            memories=debug_memories,
            graph_context=graph_facts,
            current_simulated_time=sim_now_pretty,
            system_prompt=system_prompt,
        )

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        latency_ms=latency_ms,
        memory_debug=debug_info,
    )


async def _background_memory_pipeline(
    user_id: str,
    session_id: UUID,
    user_message: str,
    assistant_response: str,
    turn_index: int,
    llm_client: LLMClient,
    episodic_store: EpisodicMemoryStore,
    knowledge_graph: KnowledgeGraph,
    entity_extractor: EntityExtractor,
    surprise_scorer: SurpriseScorer,
    contradiction_detector: ContradictionDetector,
    procedural_store: ProceduralMemoryStore,
    recent_turns: list,
    sim_now: datetime,
    sim_now_iso: str
):
    """
    Background memory pipeline — runs after response is sent.

    Pipeline:
    1. Extract facts (LangChain structured output)
    2. Surprise-gate each fact (Titans-inspired novelty check)
    3. Detect contradictions (embedding + LLM verification)
    4. Store surviving facts in episodic store
    5. Extract entity triples and update knowledge graph
    """
    try:
        # Step 1: Extract memories via LangChain
        extractor = MemoryExtractor(llm_client)
        extraction = await extractor.extract_from_turn(
            user_message=user_message,
            assistant_response=assistant_response,
            turn_index=turn_index,
            session_id=session_id,
        )

        if not extraction.memories:
            return

        # Step 2: Surprise-gate — only store novel information
        existing_memories, existing_ids = await episodic_store.get_recent_contents_with_ids(
            user_id=user_id, limit=50
        )

        stored_count = 0
        for mem in extraction.memories:
            surprise = surprise_scorer.compute_surprise_from_text(
                mem.content, existing_memories
            )

            if not surprise_scorer.should_store(surprise, mem.importance):
                logger.info(
                    "memory_surprise_gated",
                    extra={
                        "content": mem.content[:50],
                        "surprise": round(surprise, 3),
                        "importance": mem.importance,
                    },
                )
                continue

            # Step 3: Check for contradictions (with IDs for invalidation)
            contradictions = await contradiction_detector.detect(
                new_content=mem.content,
                existing_contents=existing_memories,
                existing_ids=existing_ids,
            )

            for contradiction in contradictions:
                if contradiction.old_memory_id:
                    await episodic_store.invalidate_memory(
                        memory_id=contradiction.old_memory_id,
                    )
                    logger.info(
                        "memory_contradiction_resolved",
                        extra={
                            "old": contradiction.old_content[:50],
                            "new": contradiction.new_content[:50],
                        },
                    )

            # Step 4: Store in episodic
            await episodic_store.store(
                user_id=user_id,
                content=mem.content,
                memory_type=mem.memory_type,
                importance=mem.importance,
                source_turn=turn_index,
                session_id=session_id,
                created_at=sim_now,
            )
            stored_count += 1
            existing_memories.append(mem.content)

        # Step 5: Extract entity triples and update Neo4j
        try:
            # Provide conversation history as context for better extraction
            context_str = "\n".join(f"{t['role']}: {t['content']}" for t in recent_turns[-3:])
            triples = await entity_extractor.extract_triples(
                user_message=user_message,
                assistant_response=assistant_response,
                context=context_str
            )
            if triples:
                await knowledge_graph.add_triples(
                    user_id=user_id,
                    triples=triples,
                    now_iso=sim_now_iso,
                )
        except Exception as e:
            logger.warning("graph_update_error", extra={"error": str(e)})

        logger.info(
            "background_pipeline_complete",
            extra={
                "extracted": len(extraction.memories),
                "stored": stored_count,
                "gated": len(extraction.memories) - stored_count,
            },
        )

        # Step 6: Periodically update procedural memory
        try:
            if await procedural_store.should_update(user_id):
                await procedural_store.extract_and_store(
                    user_id=user_id,
                    recent_turns=recent_turns,
                )
        except Exception as e:
            logger.warning("procedural_update_error", extra={"error": str(e)})

    except Exception as e:
        logger.error("background_pipeline_error", extra={"error": str(e)})
