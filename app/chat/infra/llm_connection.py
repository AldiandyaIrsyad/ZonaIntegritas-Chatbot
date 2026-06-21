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
    ) -> AsyncIterator[str]:
        logger.debug("chat.llm.stream_start", model=model, message_count=len(messages), max_tokens=max_tokens)
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
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

    async def close(self) -> None:
        await self._client.close()
