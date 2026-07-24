"""LLM gateway connection adapter.

Infra adapter for the chat bounded context. Wraps the OpenAI Python SDK's
async client to talk to any OpenAI-compatible backend (vLLM, Ollama,
OpenRouter, etc.).

Fulfills:
    - ``app/chat/domain/interfaces.py::ILLMConnection`` (chat generation)
    - ``app/thesis/ivm/interfaces.py::ILLMJudgeConnection`` (judge LLM —
      the same adapter satisfies this narrower port)

Wired in: ``app/chat/dependency.py::get_llm_connection``.
"""

from typing import AsyncIterator, Optional, List, Dict
import structlog
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic import SecretStr

from app.chat.domain.interfaces import ILLMConnection

logger = structlog.get_logger(__name__)


class LLMConnection(ILLMConnection):
    """Async streaming adapter for an OpenAI-compatible LLM backend.

    Fulfills: ``app/chat/domain/interfaces.py::ILLMConnection``.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Configure the underlying ``AsyncOpenAI`` client.

        Args:
            base_url: Base URL of the OpenAI-compatible backend. ``None``
                falls back to the OpenAI SDK's own default.
            api_key: API key/secret for the backend. If omitted, a dummy
                placeholder token is used (fine for backends like Ollama
                that don't check it).
            default_headers: Extra HTTP headers sent with every request
                (e.g. an OpenRouter attribution header).
        """
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

    @staticmethod
    def _suppress_thinking(model: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Disable Qwen3's thinking mode via the ``/no_think`` soft switch.

        OpenRouter's ``reasoning={"enabled": False}`` is silently ignored for
        Qwen3 on the served providers (measured: ``content`` returns empty and
        the whole answer arrives in the ``reasoning`` field), so the
        content-empty fallback in ``stream_chat``/``generate`` would stream a
        1000+ char English chain-of-thought preamble as the answer — the
        contamination Exp4 measured at 63–93%. Qwen3's documented soft switch,
        the literal ``/no_think`` token appended to the prompt, is what actually
        disables it. Scoped to ``qwen`` models so a stray control token never
        reaches a backend that would echo it verbatim.

        Args:
            model: The target model id.
            messages: The chat messages about to be sent.

        Returns:
            The messages, with ``/no_think`` appended to the last user turn for
            Qwen models; unchanged otherwise.
        """
        if "qwen" not in model.lower():
            return messages
        patched = list(messages)
        for i in range(len(patched) - 1, -1, -1):
            if patched[i].get("role") == "user":
                content = patched[i].get("content", "")
                if "/no_think" not in content:
                    patched[i] = {**patched[i], "content": f"{content.rstrip()} /no_think"}
                break
        return patched

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
        messages = self._suppress_thinking(model, messages)
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
        messages = self._suppress_thinking(model, messages)
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
        """Release the underlying ``AsyncOpenAI`` HTTP client."""
        await self._client.close()
