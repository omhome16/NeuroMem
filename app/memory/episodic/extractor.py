"""
Memory Extraction Pipeline — LangChain-powered.

After each user turn, uses LangChain to extract structured facts
(content, importance, type) from the conversation. Uses Pydantic
models with LangChain's structured output for reliable JSON parsing.
"""
import json
import logging
from typing import List, Optional
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.models.memory import ExtractedMemory, ExtractionResult, MemoryType

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Pydantic Schema for Structured Extraction ────────────────────────

class MemoryItem(BaseModel):
    """A single extracted memory from a conversation turn."""
    content: str = Field(description="A factual statement about the user")
    importance: float = Field(
        ge=0.0, le=1.0,
        description="How important this fact is (0.0 = trivial, 1.0 = critical)"
    )
    type: str = Field(
        description="Memory type: 'fact', 'preference', 'event', or 'relationship'"
    )


class ExtractionResponse(BaseModel):
    """Structured response from the memory extraction LLM chain."""
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="List of extracted memories from the conversation turn"
    )


# ── System Prompt ────────────────────────────────────────────────────

EXTRACT_SYSTEM_PROMPT = """You are a memory extraction engine for an AI assistant.
Your job is to identify high-signal, factual memories from conversation turns.

Extract ONLY:
- Explicit facts about the user (name, age, location, profession)
- Clear preferences ("I prefer X", "I hate Y", "I always do Z")
- Important events ("I got married", "I started a new job", "I moved to X")
- Relationship facts ("My wife is named Sarah", "My manager is Tom")

Do NOT extract:
- Chitchat or small talk
- Questions the user asked
- Temporary emotional states
- Ambiguous statements

Memory types: "fact", "preference", "event", "relationship"
Importance scale: 0.0 (trivial) → 1.0 (highly important, permanent)
If no memories found, return an empty memories list."""

EXTRACTION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_SYSTEM_PROMPT),
    ("human", """{context}

USER TURN:
{user_message}

ASSISTANT REPLY:
{assistant_response}

Extract memories from the conversation above. Focus on:
1. Facts or preferences mentioned by the USER.
2. Commitments, promises, or factual conclusions made by the ASSISTANT that the system should remember for the future."""),
])


class MemoryExtractor:
    """
    Extracts structured memories from conversation turns using LangChain.
    Attempts structured output first, falls back to raw JSON parsing.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def extract_from_turn(
        self,
        user_message: str,
        assistant_response: str,
        turn_index: int,
        session_id: UUID,
        conversation_context: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract memories from a single conversation turn using LangChain.

        First attempts structured output via with_structured_output().
        Falls back to raw completion + JSON parsing if structured fails.
        """
        context = f"PRIOR CONVERSATION CONTEXT:\n{conversation_context}" if conversation_context else ""

        # Attempt 1: LangChain structured output
        structured_result = await self.llm.structured_output(
            user_content=EXTRACTION_TEMPLATE.format_messages(
                context=context,
                user_message=user_message,
                assistant_response=assistant_response,
            )[-1].content,  # Get the formatted human message
            output_schema=ExtractionResponse,
            system=EXTRACT_SYSTEM_PROMPT,
            temperature=0.0,
        )

        if structured_result and isinstance(structured_result, ExtractionResponse):
            memories = self._convert_structured(structured_result)
        else:
            # Fallback: raw completion + manual JSON parsing
            logger.info("structured_extraction_fallback")
            memories = await self._extract_raw(
                user_message, assistant_response, context
            )

        logger.info(
            "memory_extraction_complete",
            extra={"turn_index": turn_index, "memories_found": len(memories)},
        )

        return ExtractionResult(
            memories=memories,
            turn_index=turn_index,
            session_id=session_id,
        )

    def _convert_structured(self, result: ExtractionResponse) -> List[ExtractedMemory]:
        """Convert LangChain structured output to our ExtractedMemory models."""
        memories = []
        for item in result.memories:
            try:
                memories.append(
                    ExtractedMemory(
                        content=item.content,
                        importance=item.importance,
                        memory_type=MemoryType(item.type),
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning("structured_memory_skip", extra={"error": str(e)})
        return memories

    async def _extract_raw(
        self,
        user_message: str,
        assistant_response: str,
        context: str,
    ) -> List[ExtractedMemory]:
        """Fallback: raw LLM completion with manual JSON parsing."""
        prompt = f"""{context}

USER TURN:
{user_message}

ASSISTANT REPLY:
{assistant_response}

Extract memories from the USER TURN only.
Return ONLY valid JSON: {{"memories": [{{"content": "...", "importance": 0.5, "type": "fact"}}]}}"""

        raw_response = await self.llm.complete(
            prompt,
            system=EXTRACT_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return self._parse_extraction_response(raw_response)

    def _parse_extraction_response(self, raw: str) -> List[ExtractedMemory]:
        """Parse raw LLM JSON response into ExtractedMemory objects."""
        import re
        import json
        try:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
            else:
                start = raw.find('{')
                end = raw.rfind('}')
                cleaned = raw[start:end+1] if start != -1 and end != -1 else raw.strip()
                
            data = json.loads(cleaned)
            memories = []
            for m in data.get("memories", []):
                try:
                    memories.append(
                        ExtractedMemory(
                            content=m["content"],
                            importance=float(m.get("importance", 0.5)),
                            memory_type=MemoryType(m.get("type", "fact")),
                        )
                    )
                except (KeyError, ValueError) as e:
                    logger.warning("memory_parse_skip", extra={"error": str(e)})
            return memories
        except Exception as e:
            logger.error("memory_extraction_json_error", extra={"error": str(e), "raw": raw[:200]})
            return []
