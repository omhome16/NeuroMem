"""
Provider-agnostic LLM client using LangChain.

Uses ChatOpenAI with a configurable base_url, which means it works with
ANY OpenAI-compatible API out of the box:
  - Groq:       https://api.groq.com/openai/v1
  - OpenAI:     https://api.openai.com/v1
  - Ollama:     http://localhost:11434/v1
  - Together:   https://api.together.xyz/v1
  - LM Studio:  http://localhost:1234/v1
  - vLLM:       http://localhost:8000/v1

Just set LLM_BASE_URL and LLM_API_KEY in your .env file.
"""
import logging
from typing import Optional, List, Type

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIConnectionError, InternalServerError

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    """
    LangChain-based LLM client wrapper.
    Compatible with any OpenAI-API-compatible provider via base_url.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        self.str_parser = StrOutputParser()

    def get_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        """Get a ChatOpenAI instance with optional overrides."""
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if kwargs:
            return self.llm.bind(**kwargs) if not kwargs.get("temperature") else ChatOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
        return self.llm

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)))
    async def complete(
        self,
        user_content: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Single-turn LLM completion using LangChain.

        Args:
            user_content: The user message / prompt
            system: Optional system prompt
            model: Model override (defaults to LLM_MODEL env var)
            temperature: Temperature override
            max_tokens: Max tokens override

        Returns:
            The assistant's response text
        """
        llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=model or settings.llm_model,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=user_content))

        response = await llm.ainvoke(messages)
        return response.content or ""

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)))
    async def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Multi-turn chat completion with optional system prompt.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            system: Optional system prompt (prepended)
            model: Model override

        Returns:
            The assistant's response text
        """
        llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=model or settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        lc_messages = []
        if system:
            lc_messages.append(SystemMessage(content=system))

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        response = await llm.ainvoke(lc_messages)
        return response.content or ""

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)))
    async def structured_output(
        self,
        user_content: str,
        output_schema: Type[BaseModel],
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Optional[BaseModel]:
        """
        Get structured (Pydantic) output from the LLM using LangChain's
        with_structured_output().

        Args:
            user_content: The user message / prompt
            output_schema: Pydantic model class for the expected output
            system: Optional system prompt
            temperature: Temperature (default 0.0 for deterministic)

        Returns:
            Parsed Pydantic model instance, or None on failure
        """
        llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
        )

        structured_llm = llm.with_structured_output(output_schema, method="function_calling")

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=user_content))

        try:
            result = await structured_llm.ainvoke(messages)
            return result
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            logger.warning(f"Network error during structured extraction, retrying... ({e})")
            raise e
        except Exception as e:
            logger.error("structured_output_failed", extra={"error": str(e)})
            return None
