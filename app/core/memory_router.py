"""
Memory Router v2 — LangGraph-powered orchestration across all four tiers.

Upgraded pipeline (10 nodes):
  route_decision     → LLM decides which tiers are relevant
  retrieve_working   → Redis (Tier 1)
  retrieve_episodic  → PostgreSQL (Tier 2)
  retrieve_semantic  → Qdrant (Tier 3)
  retrieve_graph     → Neo4j knowledge graph (Tier 4) [NEW]
  merge_results      → Combine all tier results
  rerank             → Cross-encoder re-ranking [NEW]
  token_budget_trim  → Fit within token budget [NEW]
  format_output      → Final formatting

This is the architectural showpiece — demonstrates LangGraph's
conditional branching, fan-out/fan-in, and sequential pipeline patterns.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Annotated, List, Optional, TypedDict
from uuid import UUID

from langgraph.graph import StateGraph, END

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.token_budget import TokenBudgetManager
from app.memory.episodic.store import EpisodicMemoryStore
from app.memory.graph.knowledge_graph import KnowledgeGraph
from app.memory.semantic.store import SemanticMemoryStore
from app.memory.working.working_memory import WorkingMemory
from app.models.memory import MemoryTier, MemoryType, RetrievedMemory

logger = logging.getLogger(__name__)
settings = get_settings()

ROUTER_SYSTEM_PROMPT = """You are a memory routing engine for an AI assistant.
Given the user's query, decide which memory tiers to search.

TIERS:
- working: Recent conversation context (last few turns). Use for follow-up questions, references to "it", "that", "this".
- episodic: User facts from recent sessions (last 30 days). Use for personal info, preferences, recent events.
- semantic: Long-term user profile (permanent). Use for deep personal facts, long-standing preferences, relationships.
- graph: Entity-relationship knowledge graph. Use for questions about relationships, specific people, organizations, locations.

Return ONLY JSON (no markdown fences):
{
  "tiers": ["working", "episodic", "semantic", "graph"],
  "memory_types": ["fact", "preference", "event", "relationship"],
  "query_entities": ["User", "Sarah", "Google"],
  "reasoning": "brief explanation"
}

Select only the tiers necessary. Also extract any named entities relevant to the query.
For simple chit-chat, return {"tiers": [], "memory_types": [], "query_entities": [], "reasoning": "no memory needed"}."""


# ── LangGraph State Schema ───────────────────────────────────────────

def _memory_reducer(
    existing: List[RetrievedMemory], new: List[RetrievedMemory]
) -> List[RetrievedMemory]:
    """Custom reducer that merges memory lists from parallel tier retrievals."""
    return existing + new


class RouterState(TypedDict):
    """State flowing through the LangGraph v2 memory router."""
    # Input
    query: str
    user_id: str
    session_id: str
    top_k: int
    # Routing decision
    tiers_to_query: List[str]
    memory_types: Optional[List[str]]
    query_entities: List[str]
    # Accumulated memories from each tier
    retrieved_memories: Annotated[List[RetrievedMemory], _memory_reducer]
    # Post-processing
    reranked_memories: List[RetrievedMemory]
    # Final output
    final_memories: List[RetrievedMemory]
    graph_context: List[str]
    # Simulated time for temporal awareness
    sim_now: Optional[datetime]


class MemoryRouter:
    """
    LangGraph v2 memory router — 10-node pipeline.

    Graph topology:
        route_decision
          ├── [if working]  → retrieve_working  ─┐
          ├── [if episodic] → retrieve_episodic ─┤
          ├── [if semantic] → retrieve_semantic ─┤ → merge → rerank → token_budget → format → END
          └── [if graph]    → retrieve_graph    ─┘
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        episodic_store: EpisodicMemoryStore,
        semantic_store: SemanticMemoryStore,
        knowledge_graph: KnowledgeGraph,
        llm_client: LLMClient,
        token_budget: TokenBudgetManager,
    ):
        self.working = working_memory
        self.episodic = episodic_store
        self.semantic = semantic_store
        self.graph = knowledge_graph
        self.llm = llm_client
        self.token_budget = token_budget
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph v2 StateGraph."""
        graph = StateGraph(RouterState)

        # Add 9 processing nodes
        graph.add_node("route_decision", self._route_decision_node)
        graph.add_node("retrieve_working", self._retrieve_working_node)
        graph.add_node("retrieve_episodic", self._retrieve_episodic_node)
        graph.add_node("retrieve_semantic", self._retrieve_semantic_node)
        graph.add_node("retrieve_graph", self._retrieve_graph_node)
        graph.add_node("merge_results", self._merge_results_node)
        graph.add_node("rerank", self._rerank_node)
        graph.add_node("token_budget_trim", self._token_budget_node)
        graph.add_node("format_output", self._format_output_node)

        # Entry point
        graph.set_entry_point("route_decision")

        # Conditional fan-out from route_decision
        graph.add_conditional_edges(
            "route_decision",
            self._select_retrieval_nodes,
            {
                "retrieve_working": "retrieve_working",
                "retrieve_episodic": "retrieve_episodic",
                "retrieve_semantic": "retrieve_semantic",
                "retrieve_graph": "retrieve_graph",
                "merge_results": "merge_results",
            },
        )

        # All retrieval nodes converge to merge
        graph.add_edge("retrieve_working", "merge_results")
        graph.add_edge("retrieve_episodic", "merge_results")
        graph.add_edge("retrieve_semantic", "merge_results")
        graph.add_edge("retrieve_graph", "merge_results")

        # Sequential post-processing pipeline
        graph.add_edge("merge_results", "rerank")
        graph.add_edge("rerank", "token_budget_trim")
        graph.add_edge("token_budget_trim", "format_output")
        graph.add_edge("format_output", END)

        return graph.compile()

    def _select_retrieval_nodes(self, state: RouterState) -> List[str]:
        """Conditional edge: select retrieval nodes based on routing decision."""
        tiers = state.get("tiers_to_query", [])
        if not tiers:
            return ["merge_results"]

        nodes = []
        tier_map = {
            "working": "retrieve_working",
            "episodic": "retrieve_episodic",
            "semantic": "retrieve_semantic",
            "graph": "retrieve_graph",
        }
        for tier in tiers:
            if tier in tier_map:
                nodes.append(tier_map[tier])
        return nodes if nodes else ["merge_results"]

    # ── Node Implementations ─────────────────────────────────────────

    async def _route_decision_node(self, state: RouterState) -> dict:
        """
        Node 1: Fast heuristic routing — no LLM call needed.

        Replaces the previous LLM-based router to eliminate ~800ms latency
        and one rate-limited API call from the critical path. The downstream
        cross-encoder reranker handles precision, so routing only needs to
        be directionally correct.

        Keyword signals map query intent to the appropriate memory tiers.
        Ambiguous queries fall back to querying all tiers (safe default).
        """
        query = state["query"].lower()
        tiers = ["working"]  # Always include recent conversation context
        entities = []

        # ── Relationship / Graph signals ─────────────────────────────
        relationship_signals = [
            "who is", "married", "wife", "husband", "friend", "partner",
            "works at", "manager", "boss", "colleague", "team",
            "lives in", "moved to", "located", "relationship",
            "family", "sister", "brother", "mother", "father",
            "son", "daughter", "parent", "child",
        ]
        if any(signal in query for signal in relationship_signals):
            tiers.extend(["semantic", "graph"])

        # ── Preference / Fact signals ────────────────────────────────
        fact_signals = [
            "what do", "tell me about", "remember", "know about",
            "my name", "my job", "favorite", "prefer", "like",
            "hate", "enjoy", "hobby", "hobbies", "interest",
            "allergic", "dietary", "vegetarian", "vegan",
            "what is my", "where do i", "do you know",
        ]
        if any(signal in query for signal in fact_signals):
            tiers.extend(["episodic", "semantic"])

        # ── Event / Temporal signals ─────────────────────────────────
        event_signals = [
            "when did", "last time", "recently", "yesterday",
            "last week", "planning to", "going to", "started",
            "changed", "switched", "new job", "moved",
        ]
        if any(signal in query for signal in event_signals):
            tiers.extend(["episodic", "graph"])

        # ── Default: query all tiers for ambiguous queries ───────────
        if len(tiers) == 1:
            tiers = ["working", "episodic", "semantic", "graph"]

        tiers = list(set(tiers))

        logger.info(
            "route_decision_v2",
            extra={
                "tiers": tiers,
                "entities": entities,
                "reasoning": "heuristic_keyword_routing",
            },
        )
        return {
            "tiers_to_query": tiers,
            "memory_types": None,
            "query_entities": entities,
        }

    async def _retrieve_working_node(self, state: RouterState) -> dict:
        """Node 2a: Retrieve from working memory (Redis — Tier 1)."""
        try:
            session_id = UUID(state["session_id"])
            wm_state = await self.working.get_context_for_prompt(session_id)
            results = []

            if wm_state.compressed_summary:
                results.append(
                    RetrievedMemory(
                        content=f"[Session Summary] {wm_state.compressed_summary}",
                        memory_type=MemoryType.FACT,
                        tier=MemoryTier.WORKING,
                        relevance_score=0.95,
                        importance=0.8,
                        created_at=wm_state.turns[0].timestamp if wm_state.turns else datetime.now(timezone.utc),
                    )
                )

            for turn in wm_state.turns[-4:]:
                if turn.role == "user":
                    results.append(
                        RetrievedMemory(
                            content=f"[Recent] User said: {turn.content}",
                            memory_type=MemoryType.EVENT,
                            tier=MemoryTier.WORKING,
                            relevance_score=0.8,
                            importance=0.6,
                            created_at=turn.timestamp,
                        )
                    )

            return {"retrieved_memories": results}
        except Exception as e:
            logger.error("working_retrieval_error", extra={"error": str(e)})
            return {"retrieved_memories": []}

    async def _retrieve_episodic_node(self, state: RouterState) -> dict:
        """Node 2b: Retrieve from episodic memory (PostgreSQL — Tier 2)."""
        try:
            memory_types = None
            if state.get("memory_types"):
                memory_types = [MemoryType(t) for t in state["memory_types"]]

            results = await self.episodic.retrieve(
                user_id=state["user_id"],
                query_text=state["query"],
                limit=state.get("top_k", 5),
                memory_types=memory_types,
                sim_now=state.get("sim_now"),
            )
            return {"retrieved_memories": results}
        except Exception as e:
            logger.error("episodic_retrieval_error", extra={"error": str(e)})
            return {"retrieved_memories": []}

    async def _retrieve_semantic_node(self, state: RouterState) -> dict:
        """Node 2c: Retrieve from semantic memory (Qdrant — Tier 3)."""
        try:
            memory_types = None
            if state.get("memory_types"):
                memory_types = [MemoryType(t) for t in state["memory_types"]]

            results = await self.semantic.search(
                user_id=state["user_id"],
                query_text=state["query"],
                limit=state.get("top_k", 5),
                memory_types=memory_types,
            )
            return {"retrieved_memories": results}
        except Exception as e:
            logger.error("semantic_retrieval_error", extra={"error": str(e)})
            return {"retrieved_memories": []}

    async def _retrieve_graph_node(self, state: RouterState) -> dict:
        """Node 2d: Retrieve from knowledge graph (Neo4j — Tier 4). [NEW]"""
        try:
            entities = state.get("query_entities", [])
            if not entities:
                entities = ["User"]

            facts = await self.graph.query_context(
                user_id=state["user_id"],
                query_entities=entities,
                max_hops=2,
            )

            results = [
                RetrievedMemory(
                    content=f"[Knowledge Graph] {fact}",
                    memory_type=MemoryType.RELATIONSHIP,
                    tier=MemoryTier.GRAPH,
                    relevance_score=0.85,
                    importance=0.9,
                    created_at=datetime.now(timezone.utc),
                )
                for fact in facts
            ]

            return {
                "retrieved_memories": results,
                "graph_context": facts,
            }
        except Exception as e:
            logger.error("graph_retrieval_error", extra={"error": str(e)})
            return {"retrieved_memories": [], "graph_context": []}

    async def _merge_results_node(self, state: RouterState) -> dict:
        """Node 3: Deduplicate memories from all tiers."""
        all_memories = state.get("retrieved_memories", [])

        seen = set()
        deduped = []
        for mem in all_memories:
            key = mem.content.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(mem)

        deduped.sort(key=lambda m: m.relevance_score, reverse=True)
        return {"reranked_memories": deduped}

    async def _rerank_node(self, state: RouterState) -> dict:
        """Node 4: Cross-encoder re-ranking for precision. [NEW]"""
        memories = state.get("reranked_memories", [])
        if len(memories) <= 3:
            return {"reranked_memories": memories}

        try:
            reranked = self.token_budget.rerank(
                query=state["query"],
                memories=memories,
                top_k=15,
            )
            return {"reranked_memories": reranked}
        except Exception as e:
            logger.warning("reranking_fallback", extra={"error": str(e)})
            return {"reranked_memories": memories}

    async def _token_budget_node(self, state: RouterState) -> dict:
        """Node 5: Fit memories within token budget. [NEW]"""
        memories = state.get("reranked_memories", [])
        fitted = self.token_budget.fit_to_budget(memories)
        return {"final_memories": fitted}

    async def _format_output_node(self, state: RouterState) -> dict:
        """Node 6: Final formatting pass."""
        return {"final_memories": state.get("final_memories", [])}

    # ── Public API ───────────────────────────────────────────────────

    async def route_and_retrieve(
        self,
        user_id: str,
        session_id: UUID,
        query: str,
        top_k_per_tier: int = 5,
        sim_now: Optional[datetime] = None,
    ) -> List[RetrievedMemory]:
        """
        Main retrieval entrypoint — invokes the LangGraph v2 pipeline.

        Pipeline: route → fan-out retrieval → merge → rerank → token budget → format
        """
        initial_state: RouterState = {
            "query": query,
            "user_id": user_id,
            "session_id": str(session_id),
            "top_k": top_k_per_tier,
            "tiers_to_query": [],
            "memory_types": None,
            "query_entities": [],
            "retrieved_memories": [],
            "reranked_memories": [],
            "final_memories": [],
            "graph_context": [],
            "sim_now": sim_now,
        }

        result = await self._graph.ainvoke(initial_state)
        return result.get("final_memories", [])
