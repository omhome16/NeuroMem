"""
LangGraph v2 — Adaptive Memory Consolidation.

A multi-stage pipeline for promoting episodic memories to the semantic tier:
  1. score_memories           → Ebbinghaus decay filtering
  2. cluster_by_topic         → Group remaining episodes by semantic topic
  3. extract_graph_updates    → Find entity-relationships for the knowledge graph
  4. detect_contradictions    → Invalidate old facts based on new consolidated ones
  5. compress_to_semantic     → LLM summarizes clusters into final semantic facts
  6. compute_surprise_deltas  → Measure information gain
"""
import logging
from typing import List, Dict, Optional, TypedDict, Any
from uuid import UUID
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.surprise_scorer import SurpriseScorer
from app.core.contradiction_detector import ContradictionDetector
from app.memory.episodic.scorer import EbbinghausScorer
from app.memory.graph.entity_extractor import EntityExtractor, EntityTriple
from app.models.memory import EpisodicMemory, SemanticMemory, MemoryType

logger = logging.getLogger(__name__)
settings = get_settings()


class ConsolidationState(TypedDict):
    """State flowing through the consolidation graph."""
    user_id: str
    raw_episodes: List[EpisodicMemory]
    memories_to_keep: List[EpisodicMemory]
    memories_to_delete: List[EpisodicMemory]
    clustered_episodes: Dict[str, List[EpisodicMemory]]
    semantic_facts: List[Dict[str, Any]]
    graph_triples: List[EntityTriple]
    contradictions_resolved: int
    surprise_delta: float


class SemanticFact(BaseModel):
    content: str = Field(description="A clear standalone factual sentence")
    importance: float = Field(ge=0.0, le=1.0)
    type: str = Field(description="fact, preference, event, or relationship")


class ClusterConsolidationResponse(BaseModel):
    semantic_facts: List[SemanticFact]


class ConsolidationGraph:
    """
    Multi-stage consolidation pipeline.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        entity_extractor: EntityExtractor,
        contradiction_detector: ContradictionDetector,
        surprise_scorer: SurpriseScorer,
    ):
        self.llm = llm_client
        self.entity_extractor = entity_extractor
        self.contradiction_detector = contradiction_detector
        self.surprise_scorer = surprise_scorer
        self.scorer = EbbinghausScorer()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ConsolidationState)

        graph.add_node("score_memories", self._score_node)
        graph.add_node("cluster_by_topic", self._cluster_node)
        graph.add_node("extract_graph_updates", self._graph_node)
        graph.add_node("detect_contradictions", self._contradiction_node)
        graph.add_node("compress_to_semantic", self._compress_node)
        graph.add_node("compute_surprise_deltas", self._surprise_node)

        graph.set_entry_point("score_memories")

        graph.add_edge("score_memories", "cluster_by_topic")
        graph.add_edge("cluster_by_topic", "extract_graph_updates")
        graph.add_edge("extract_graph_updates", "detect_contradictions")
        graph.add_edge("detect_contradictions", "compress_to_semantic")
        graph.add_edge("compress_to_semantic", "compute_surprise_deltas")
        graph.add_edge("compute_surprise_deltas", END)

        return graph.compile()

    async def _score_node(self, state: ConsolidationState) -> dict:
        episodes = state["raw_episodes"]
        if not episodes:
            return {"memories_to_keep": [], "memories_to_delete": []}

        keep, delete_candidates = self.scorer.filter_for_deletion(
            episodes, threshold=settings.consolidation_decay_threshold
        )
        return {"memories_to_keep": keep, "memories_to_delete": delete_candidates}

    async def _cluster_node(self, state: ConsolidationState) -> dict:
        keep = state.get("memories_to_keep", [])
        if not keep:
            return {"clustered_episodes": {}}

        # Simple clustering - put all in one cluster for now to save API calls
        # A full production version would use HDBSCAN or agglomerative clustering
        return {"clustered_episodes": {"general": keep}}

    async def _graph_node(self, state: ConsolidationState) -> dict:
        keep = state.get("memories_to_keep", [])
        if not keep:
            return {"graph_triples": []}

        content_blob = "\n".join([m.content for m in keep])
        triples = await self.entity_extractor.extract_triples(
            user_message=content_blob
        )
        return {"graph_triples": triples}

    async def _contradiction_node(self, state: ConsolidationState) -> dict:
        # In a full flow, here we would compare the new clusters against the knowledge graph
        # For simplicity in this demo, we simulate the node execution
        return {"contradictions_resolved": 0}

    async def _compress_node(self, state: ConsolidationState) -> dict:
        clusters = state.get("clustered_episodes", {})
        all_facts = []

        system_prompt = """You are a memory consolidation engine.
Distill the given episodic memories into 5-10 core semantic facts.
Merge redundant facts and resolve contradictions."""

        for cluster_id, episodes in clusters.items():
            if not episodes:
                continue

            memory_list = "\n".join([f"- {m.content}" for m in episodes])
            prompt = f"EPISODIC MEMORIES:\n{memory_list}\n\nDistill into core semantic facts."

            result = await self.llm.structured_output(
                user_content=prompt,
                output_schema=ClusterConsolidationResponse,
                system=system_prompt,
                temperature=0.0,
            )

            if result and isinstance(result, ClusterConsolidationResponse):
                for fact in result.semantic_facts:
                    all_facts.append({
                        "content": fact.content,
                        "importance": fact.importance,
                        "type": fact.type,
                        "source_ids": [str(m.id) for m in episodes]
                    })

        return {"semantic_facts": all_facts}

    async def _surprise_node(self, state: ConsolidationState) -> dict:
        facts = state.get("semantic_facts", [])
        # Mock calculation for pipeline completeness
        delta = 0.5 if facts else 0.0
        return {"surprise_delta": delta}

    async def run(self, user_id: str, episodes: List[EpisodicMemory]) -> ConsolidationState:
        initial_state: ConsolidationState = {
            "user_id": user_id,
            "raw_episodes": episodes,
            "memories_to_keep": [],
            "memories_to_delete": [],
            "clustered_episodes": {},
            "semantic_facts": [],
            "graph_triples": [],
            "contradictions_resolved": 0,
            "surprise_delta": 0.0
        }
        result = await self._graph.ainvoke(initial_state)
        return result
