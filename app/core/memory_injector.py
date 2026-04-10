"""
Memory Injector — Builds the system prompt with tiered memories.

Structures retrieved memories into a clear hierarchy:
  - Procedural: how the user likes to interact (style, preferences)
  - Long-term (semantic): permanent user profile
  - Recent (episodic): last 30 days of facts
  - This conversation (working): current session context
"""
from typing import List, Optional

from app.models.memory import RetrievedMemory, MemoryTier


class MemoryInjector:
    """
    Builds the system prompt with injected memories, structured by tier.
    Procedural memory (Tier 5) is injected as the highest-priority section,
    shaping the AI's communication style before any factual content.
    """

    SYSTEM_TEMPLATE = """You are a helpful, personalized AI assistant with memory of past interactions.

{time_block}
{procedural_block}
{memory_block}

Answer the user's question naturally. Use the memories when they are directly relevant.
Do NOT mention the memory system or that you have stored memories. Act as if you simply know these things."""

    def build_system_prompt(
        self,
        memories: List[RetrievedMemory],
        procedural_context: Optional[str] = None,
        base_system: Optional[str] = None,
        current_time: Optional[str] = None,
    ) -> str:
        """
        Inject memories into the system prompt.

        Format:
            ## How You Interact With This User (Procedural)
            [communication style, preferences, patterns]

            ## What I Know About You (Long-Term)
            [semantic facts — permanent profile]

            ## Recent Context (Last 30 Days)
            [episodic facts — last 30 days]

            ## This Conversation
            [working memory — current session]
        """
        # ── Procedural block (how to interact) ───────────────────────
        if procedural_context:
            procedural_block = (
                f"## How You Interact With This User\n{procedural_context}"
            )
        else:
            procedural_block = ""

        # ── Memory blocks (what to remember) ─────────────────────────
        if not memories:
            memory_block = "(No memories found — treat this as a first interaction.)"
        else:
            semantic = [m for m in memories if m.tier == MemoryTier.SEMANTIC]
            episodic = [m for m in memories if m.tier == MemoryTier.EPISODIC]
            working = [m for m in memories if m.tier == MemoryTier.WORKING]
            graph = [m for m in memories if m.tier == MemoryTier.GRAPH]

            sections = []

            if graph:
                section = "## Relationships & Knowledge Graph\n"
                section += "\n".join(f"- {m.content}" for m in graph)
                sections.append(section)

            if semantic:
                section = "## What I Know About You (Long-Term)\n"
                section += "\n".join(f"- {m.content}" for m in semantic)
                sections.append(section)

            if episodic:
                section = "## Recent Context (Last 30 Days)\n"
                section += "\n".join(f"- {m.content}" for m in episodic)
                sections.append(section)

            if working:
                section = "## This Conversation\n"
                section += "\n".join(f"- {m.content}" for m in working)
                sections.append(section)

            memory_block = "\n\n".join(sections) if sections else "(No relevant memories found.)"

        # ── Time block (orientation) ─────────────────────────────────
        time_block = f"## Current Simulated Time\n{current_time}" if current_time else ""

        base = base_system or ""
        full_system = self.SYSTEM_TEMPLATE.format(
            time_block=time_block,
            procedural_block=procedural_block,
            memory_block=memory_block,
        )
        if base:
            full_system = base + "\n\n" + full_system
        return full_system

