"""HTTP clients for evaluation scripts.

Provides minimal adapters to call the Infinity inference server and
OpenRouter LLM API without importing from ``chat/infra`` or ``kb/infra``
(keeping eval scripts decoupled from production wiring).

These clients implement the thesis Protocol interfaces
(``ISafetyModel``, ``INLIModel``, ``IEmbeddingModel``, ``ILLMJudgeConnection``)
so they can be passed directly to ``IVMService`` and ``RAMService``.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.thesis.ivm.interfaces import SafetyResult
from app.thesis.ram.interfaces import NLIResult


class EvalSafetyClient:
    """Minimal safety model adapter calling Infinity's /classify endpoint.

    Implements ``ISafetyModel`` for use with ``IVMService`` in eval scripts.

    Args:
        base_url: Infinity server base URL.
        model: Prompt guard model identifier.
        threshold: Security threshold for malicious classification.
    """

    def __init__(self, base_url: str, model: str, threshold: float = 0.5) -> None:
        self._model = model
        self._threshold = threshold
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def check_prompt(self, text: str) -> SafetyResult:
        """Classify input text for prompt injection.

        Args:
            text: Input string to classify.

        Returns:
            SafetyResult with is_safe flag and message.
        """
        try:
            response = await self._client.post(
                "/classify",
                json={"model": self._model, "input": [text]},
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if not items:
                return SafetyResult(is_safe=True, message="No prediction")

            predictions = items[0]
            score_dict: Dict[str, float] = {}
            if isinstance(predictions, list):
                for p in predictions:
                    score_dict[str(p.get("label", "")).lower()] = float(p.get("score", 0.0))

            # Llama-PG-2 labels: safe/injection (or benign/injection)
            injection_score = (
                score_dict.get("injection", 0.0)
                or score_dict.get("unsafe", 0.0)
            )
            is_safe = injection_score < self._threshold
            return SafetyResult(
                is_safe=is_safe,
                message="malicious" if not is_safe else "safe",
            )
        except Exception as e:
            # Fail-closed: treat errors as unsafe
            return SafetyResult(is_safe=False, message=f"error: {e}")

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class EvalNLIClient:
    """Minimal NLI adapter calling Infinity's /classify endpoint.

    Implements ``INLIModel`` for use with ``RAMService`` in eval scripts.

    Args:
        base_url: Infinity server base URL.
        model: NLI model identifier.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        self._sep = " </s></s> " if "roberta" in model.lower() else " [SEP] "
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI inference on a premise/hypothesis pair.

        Args:
            premise: Reference context.
            hypothesis: Statement to verify.

        Returns:
            NLIResult with canonical label and per-class scores.
        """
        text = f"{premise}{self._sep}{hypothesis}"
        try:
            response = await self._client.post(
                "/classify",
                json={"model": self._model, "input": [text], "raw_scores": True},
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("data", [])
            if not items:
                return NLIResult(label="neutral", entailment_score=0.5)

            predictions = items[0]
            score_dict: Dict[str, float] = {}
            if isinstance(predictions, list):
                for p in predictions:
                    score_dict[str(p.get("label", "")).lower()] = float(p.get("score", 0.0))

            label_map = {
                "label_0": "entailment",
                "label_1": "neutral",
                "label_2": "contradiction",
                "entailment": "entailment",
                "neutral": "neutral",
                "contradiction": "contradiction",
            }
            scores: Dict[str, float] = {
                label_map.get(k, "neutral"): v for k, v in score_dict.items()
            }

            best_label = max(scores, key=scores.get, default="neutral")
            return NLIResult(
                label=best_label,
                entailment_score=scores.get("entailment", 0.0),
                neutral_score=scores.get("neutral", 0.0),
                contradiction_score=scores.get("contradiction", 0.0),
            )
        except Exception:
            return NLIResult(label="neutral", entailment_score=0.5)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class EvalEmbeddingClient:
    """Minimal embedding adapter calling Infinity's /embeddings endpoint.

    Implements ``IEmbeddingModel`` for use with ``RAMService`` in eval scripts.

    Args:
        base_url: Infinity server base URL.
        model: Embedding model identifier.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a list of texts.

        Args:
            texts: Input strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        response = await self._client.post(
            "/embeddings",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data.get("data", [])]

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class EvalLLMClient:
    """OpenAI-compatible LLM client for eval scripts (LLM Generator + Judge).

    Connects to OpenRouter or any OpenAI-compatible API. Used for:
    - Prompting-based safety baseline (Exp 1a)
    - LLM-as-Judge relevance evaluation (Exp 1b)
    - End-to-end generation (Exp 4)

    Args:
        base_url: API base URL.
        api_key: API key.
        model: Default model identifier.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request and return the full response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Override model (defaults to self._model).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": model or self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 100,
    ) -> AsyncIterator[str]:
        """Stream a chat completion (implements ILLMJudgeConnection).

        Args:
            model: Model identifier.
            messages: List of message dicts.
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they arrive.
        """
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": 0.0,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def get_llm_client_from_env() -> EvalLLMClient:
    """Create an EvalLLMClient from environment variables.

    Reads:
        OPENROUTER_API_KEY: API key for OpenRouter.
        EVAL_LLM_MODEL: Model identifier (default: deepseek/deepseek-chat).

    Returns:
        Configured EvalLLMClient.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required for LLM eval."
        )
    model = os.environ.get("EVAL_LLM_MODEL", "deepseek/deepseek-chat")
    return EvalLLMClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
    )
