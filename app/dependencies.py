"""
Dependency Injection container.

Provides singleton instances of all core components.
Uses FastAPI's Depends() for clean constructor injection.
"""
import logging
from functools import lru_cache
from typing import Optional

import redis.asyncio as aioredis

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.surprise_scorer import SurpriseScorer
from app.core.contradiction_detector import ContradictionDetector
from app.core.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singletons ───────────────────────────────────────────────────────

_llm_client: Optional[LLMClient] = None
_surprise_scorer: Optional[SurpriseScorer] = None
_contradiction_detector: Optional[ContradictionDetector] = None
_token_budget: Optional[TokenBudgetManager] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_token_budget() -> TokenBudgetManager:
    global _token_budget
    if _token_budget is None:
        _token_budget = TokenBudgetManager()
    return _token_budget


def get_surprise_scorer() -> SurpriseScorer:
    global _surprise_scorer
    if _surprise_scorer is None:
        from app.memory.semantic.embedder import Embedder
        _surprise_scorer = SurpriseScorer(embedder=Embedder())
    return _surprise_scorer


def get_contradiction_detector() -> ContradictionDetector:
    global _contradiction_detector
    if _contradiction_detector is None:
        from app.memory.semantic.embedder import Embedder
        _contradiction_detector = ContradictionDetector(
            embedder=Embedder(),
            llm_client=get_llm_client(),
        )
    return _contradiction_detector


# ── Store Dependencies (require async clients) ──────────────────────

async def get_working_memory():
    from app.db.redis_client import get_redis_client
    redis = await get_redis_client()
    from app.memory.working.working_memory import WorkingMemory
    return WorkingMemory(redis_client=redis, llm_client=get_llm_client())


async def get_episodic_store():
    from app.db.postgres import get_pg_pool
    from app.db.redis_client import get_redis_client
    from app.memory.episodic.store import EpisodicMemoryStore
    pg = await get_pg_pool()
    redis = await get_redis_client()
    return EpisodicMemoryStore(pg_pool=pg, redis_client=redis)


async def get_semantic_store():
    from app.db.qdrant_client import get_qdrant_client
    from app.memory.semantic.store import SemanticMemoryStore
    from app.memory.semantic.embedder import Embedder
    qdrant = get_qdrant_client()
    return SemanticMemoryStore(client=qdrant, embedder=Embedder())


async def get_knowledge_graph():
    from app.db.neo4j_client import get_neo4j_driver
    from app.memory.graph.knowledge_graph import KnowledgeGraph
    driver = get_neo4j_driver()
    return KnowledgeGraph(driver=driver)


async def get_entity_extractor():
    from app.memory.graph.entity_extractor import EntityExtractor
    return EntityExtractor(llm_client=get_llm_client())


async def get_procedural_store():
    from app.db.redis_client import get_redis_client
    from app.memory.procedural.store import ProceduralMemoryStore
    redis = await get_redis_client()
    return ProceduralMemoryStore(redis_client=redis, llm_client=get_llm_client())


async def get_memory_router():
    from app.core.memory_router import MemoryRouter
    working = await get_working_memory()
    episodic = await get_episodic_store()
    semantic = await get_semantic_store()
    graph = await get_knowledge_graph()
    return MemoryRouter(
        working_memory=working,
        episodic_store=episodic,
        semantic_store=semantic,
        knowledge_graph=graph,
        llm_client=get_llm_client(),
        token_budget=get_token_budget(),
    )
