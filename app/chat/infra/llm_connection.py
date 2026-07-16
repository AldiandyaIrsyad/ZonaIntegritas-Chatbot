"""LLM gateway connection adapter."""

from typing import AsyncIterator, Optional, List, Dict
import structlog
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic import SecretStr

from app.chat.domain.interfaces import ILLMConnection

logger = structlog.get_logger(__name__)

class LLMConnection(ILLMConnection):
    """Async streaming adapter for an OpenAI-compatible LLM backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=default_headers,
        )
        logger.info(
            "chat.llm.initialized",
            base_url=base_url,
            has_api_key=api_key is not None,
        )

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, handling reasoning models.

        Reasoning models (e.g. Gemini 2.5 Flash, GLM 5.2) may return
        `content: null` with the actual text in a `reasoning` field.
        We disable reasoning via `extra_body` when possible and fall
        back to the `reasoning` delta when `content` is absent.

        Args:
            model: Model identifier.
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.

        Yields:
            Response text chunks.
        """
        logger.debug("chat.llm.stream_start", model=model, message_count=len(messages), max_tokens=max_tokens, temperature=temperature)
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,  
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                extra_body={"reasoning": {"enabled": False}},
            )
            async for chunk in stream:  # type: ignore[union-attr]
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Primary: content field (non-reasoning models or reasoning disabled)
                content = getattr(delta, "content", None)
                if content:
                    yield content
                else:
                    # Fallback: reasoning field (models where reasoning can't be disabled)
                    reasoning = getattr(delta, "reasoning", None)
                    if reasoning:
                        yield reasoning
        except APIConnectionError as exc:
            logger.error("chat.llm.connection_error", model=model, error=str(exc))
            raise
        except APIError as exc:
            logger.error("chat.llm.api_error", model=model, status_code=getattr(exc, "status_code", None), error=str(exc))
            raise
        except Exception as exc:
            logger.error("chat.llm.unexpected_error", model=model, error=str(exc))
            raise

        logger.debug("chat.llm.stream_end", model=model)

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """Generate a complete (non-streaming) chat completion.

        Used for tasks requiring the full response before proceeding
        (e.g. HyDE hypothetical document generation). Uses ``stream=False``
        on the same AsyncOpenAI client.

        Args:
            model: Model identifier.
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.

        Returns:
            The full response text.
        """
        logger.debug(
            "chat.llm.generate_start",
            model=model,
            message_count=len(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                extra_body={"reasoning": {"enabled": False}},
            )
            content = response.choices[0].message.content  # type: ignore[union-attr]
            if content:
                return content
            # Fallback: reasoning field (models where reasoning can't be disabled)
            reasoning = getattr(response.choices[0].message, "reasoning", None)  # type: ignore[union-attr]
            return reasoning or ""
        except APIConnectionError as exc:
            logger.error("chat.llm.connection_error", model=model, error=str(exc))
            raise
        except APIError as exc:
            logger.error(
                "chat.llm.api_error",
                model=model,
                status_code=getattr(exc, "status_code", None),
                error=str(exc),
            )
            raise
        except Exception as exc:
            logger.error("chat.llm.unexpected_error", model=model, error=str(exc))
            raise

    async def close(self) -> None:
        await self._client.close()
