"""
Core data models for the NeuroMem memory system.

Defines the Pydantic v2 schemas for all three memory tiers,
extraction results, conflict resolution, and query results.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    RELATIONSHIP = "relationship"


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    GRAPH = "graph"


# ── Extracted Memory (from LLM pipeline) ─────────────────────────────

class ExtractedMemory(BaseModel):
    """A single fact/preference/event extracted by the LLM from a turn."""
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    memory_type: MemoryType
    tags: List[str] = []


class ExtractionResult(BaseModel):
    """Result of extracting memories from a conversation turn."""
    memories: List[ExtractedMemory]
    turn_index: int
    session_id: UUID


# ── Episodic Memory (stored in PostgreSQL — Tier 2) ─────────────────

class EpisodicMemory(BaseModel):
    """A memory stored in the episodic (mid-term) store."""
    id: UUID
    user_id: str
    content: str
    memory_type: MemoryType
    importance: float
    recall_count: int
    tags: List[str]
    source_turn: Optional[int]
    session_id: Optional[UUID]
    created_at: datetime
    last_recalled: Optional[datetime]
    consolidated: bool
    decay_score: float


# ── Semantic Memory (stored in Qdrant — Tier 3) ─────────────────────

class SemanticMemory(BaseModel):
    """A consolidated long-term memory in the vector store."""
    id: str  # Qdrant point ID (UUID as string)
    user_id: str
    content: str
    memory_type: MemoryType
    importance: float
    embedding: Optional[List[float]] = None
    created_at: datetime
    source_episode_ids: List[str] = []


# ── Working Memory Turn (Redis — Tier 1) ─────────────────────────────

class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    turn_index: int


class WorkingMemoryState(BaseModel):
    """Full state of working memory for a session."""
    session_id: UUID
    user_id: str
    turns: List[ConversationTurn]
    compressed_summary: Optional[str] = None


# ── Memory Query Result ──────────────────────────────────────────────

class RetrievedMemory(BaseModel):
    """A memory retrieved from any tier, ready for prompt injection."""
    content: str
    memory_type: MemoryType
    tier: MemoryTier
    relevance_score: float
    importance: float
    created_at: datetime
