"""
Entity-Relationship Extractor for the Temporal Knowledge Graph.

Uses LangChain structured output to extract (subject, predicate, object)
triples from conversation turns. These triples form the edges of the
user's personal knowledge graph.

Example extraction:
    "My wife Sarah works as a doctor at Apollo Hospital"
    → (User, married_to, Sarah)
    → (Sarah, works_as, doctor)
    → (Sarah, works_at, Apollo Hospital)
"""
import logging
from typing import List
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Pydantic Schemas for Structured Extraction ───────────────────────

class EntityTriple(BaseModel):
    """A single entity-relationship triple."""
    subject: str = Field(description="The subject entity (e.g., 'User', 'Sarah')")
    predicate: str = Field(description="The relationship (e.g., 'lives_in', 'works_at', 'married_to')")
    object: str = Field(description="The object entity (e.g., 'Pune', 'Google', 'Sarah')")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Extraction confidence")


class EntityExtractionResponse(BaseModel):
    """Structured response from entity extraction."""
    entities: List[EntityTriple] = Field(
        default_factory=list,
        description="List of entity-relationship triples extracted from the text"
    )


ENTITY_SYSTEM_PROMPT = """You are an entity-relationship extraction engine for a personal knowledge graph.

Extract structured (subject, predicate, object) triples from the user's message.

Rules:
- Use "User" as the subject when the user talks about themselves
- Use concise, snake_case predicates: lives_in, works_at, married_to, prefers, has_pet, allergic_to, age_is, etc.
- Extract ONLY factual relationships, not opinions or emotions
- Each triple should be a standalone fact
- If a person is mentioned, extract their relationships too

Examples:
  "I'm a software engineer at Google" → (User, works_as, software engineer), (User, works_at, Google)
  "My wife Sarah is a doctor" → (User, married_to, Sarah), (Sarah, works_as, doctor)
  "I moved to Pune from Mumbai" → (User, lives_in, Pune), (User, previously_lived_in, Mumbai)

If no entities can be extracted, return an empty list."""


class EntityExtractor:
    """
    Extracts entity-relationship triples from conversation turns
    using LangChain structured output.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def extract_triples(
        self,
        user_message: str,
        assistant_response: str = "",
        context: str = "",
    ) -> List[EntityTriple]:
        """
        Extract entity-relationship triples from a conversation turn.

        Uses LangChain with_structured_output() for reliable parsing.
        Falls back to raw completion if structured output fails.
        """
        prompt = f"""CONVERSATION CONTEXT: {context}
USER MESSAGE: {user_message}
ASSISTANT REPLY: {assistant_response}

Extract entity-relationship triples from the conversation above.
Include facts mentioned by the User AND commitments or factual conclusions made by the Assistant."""

        # Primary: structured output
        result = await self.llm.structured_output(
            user_content=prompt,
            output_schema=EntityExtractionResponse,
            system=ENTITY_SYSTEM_PROMPT,
            temperature=0.0,
        )

        if result and isinstance(result, EntityExtractionResponse):
            triples = result.entities
            logger.info(
                "entities_extracted",
                extra={"count": len(triples)},
            )
            return triples

        # Fallback: raw extraction
        logger.info("entity_extraction_fallback")
        return await self._extract_raw(user_message)

    async def _extract_raw(self, user_message: str) -> List[EntityTriple]:
        """Fallback: raw LLM completion with JSON parsing."""
        import json
        import re

        prompt = f"""Extract entity-relationship triples from this text:
"{user_message}"

Return ONLY JSON: {{"entities": [{{"subject": "User", "predicate": "lives_in", "object": "Pune", "confidence": 0.9}}]}}"""

        raw = await self.llm.complete(prompt, system=ENTITY_SYSTEM_PROMPT, temperature=0.0)
        try:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
            else:
                start = raw.find('{')
                end = raw.rfind('}')
                cleaned = raw[start:end+1] if start != -1 and end != -1 else raw.strip()
                
            data = json.loads(cleaned)
            return [EntityTriple(**t) for t in data.get("entities", [])]
        except Exception as e:
            logger.error("entity_extraction_parse_error", extra={"error": str(e), "raw": raw[:200]})
            return []
