"""
Chat request/response models for the /chat endpoint.
"""
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""
    message: str
    session_id: Optional[UUID] = None
    user_id: str = "default"
    include_memory_debug: bool = False  # returns retrieved memories in response


class MemoryDebugInfo(BaseModel):
    """Debug information regarding retrieved memory."""
    memories_retrieved: int
    tiers_queried: List[str]
    memories: List[Dict[str, Any]]
    graph_context: List[str] = []
    current_simulated_time: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from the chat endpoint with optional memory debug info."""
    reply: str
    session_id: UUID
    latency_ms: float
    memory_debug: Optional[MemoryDebugInfo] = None
