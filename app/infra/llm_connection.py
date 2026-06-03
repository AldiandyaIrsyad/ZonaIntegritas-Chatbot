"""LLM gateway connection adapter.

Manages the low-level async streaming connection to the LLM backend — either
OpenRouter (production) or a local Ollama container — using the
OpenAI-compatible ``/chat/completions`` endpoint.

Text is always streamed; blocking completion calls are never used so the HTTP
server can begin flushing response tokens immediately.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

import structlog
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic import SecretStr

logger = structlog.get_logger(__name__)


class LLMConnection:
    """Async streaming adapter for an OpenAI-compatible LLM backend.

    Supports OpenRouter and Ollama transparently — both expose the same
    ``/chat/completions`` interface.  Satisfies the
    :class:`~app.core.interfaces.infra.ILLMConnection` Protocol structurally.

    Args:
        base_url: Provider endpoint base URL.  Pass ``None`` to use the
                  default OpenAI URL.
        api_key: Authentication key.  If ``None``, a placeholder token is
                 used — Ollama ignores authentication.
        default_headers: HTTP headers attached to every request
                         (e.g. ``{"X-Title": "chatbot"}`` for OpenRouter
                         rate-limit attribution).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=default_headers,
        )
        logger.info(
            "LLMConnection initialised",
            base_url=base_url,
            has_api_key=api_key is not None,
        )

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from the LLM backend.

        Opens a server-sent-event stream and yields non-empty text delta
        chunks as they arrive.  The caller assembles chunks into a complete
        response.

        Args:
            model: Model identifier (e.g. ``"openai/gpt-4o"`` for OpenRouter
                   or ``"llama3.1:8b"`` for Ollama).
            messages: Conversation history in OpenAI message format
                      (``[{"role": "user", "content": "..."}]``).
            max_tokens: Maximum number of tokens the model may generate.

        Yields:
            Non-empty text delta strings produced by the model.

        Raises:
            APIConnectionError: If the connection to the backend cannot be
                established.
            APIError: If the backend returns an error response.
            Exception: Re-raised for any other unexpected error.
        """
        logger.debug(
            "llm.stream.start",
            model=model,
            message_count=len(messages),
            max_tokens=max_tokens,
        )
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIConnectionError as exc:
            logger.error(
                "llm.stream.connection_error",
                model=model,
                error=str(exc),
            )
            raise
        except APIError as exc:
            logger.error(
                "llm.stream.api_error",
                model=model,
                status_code=exc.status_code,
                error=str(exc),
            )
            raise
        except Exception as exc:
            logger.error(
                "llm.stream.unexpected_error",
                model=model,
                error=str(exc),
            )
            raise

        logger.debug("llm.stream.end", model=model)

    async def close(self) -> None:
        """Close the underlying OpenAI async HTTP client."""
        await self._client.close()