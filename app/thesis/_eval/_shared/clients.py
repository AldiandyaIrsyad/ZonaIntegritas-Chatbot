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
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tokenizers import Tokenizer

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

            # Llama-Prompt-Guard-2-86M's actual Infinity /classify labels are
            # "MALICIOUS"/"BENIGN" (confirmed against a live call — LABEL_0/
            # LABEL_1 are also handled as a fallback, matching production's
            # PromptGuardClient). The previous "injection"/"unsafe" guesses
            # never matched any real label, so injection_score was always
            # 0.0 and every sample was classified safe regardless of input —
            # this eval client, not the model, was silently broken.
            injection_score = (
                score_dict.get("malicious", 0.0)
                or score_dict.get("label_1", 0.0)
                or score_dict.get("injection", 0.0)
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


# StevenLimcorn/indo-roberta-indonli is a RoBERTa variant with a 512-token
# position-embedding table and no server-side truncation configured on
# Infinity (confirmed: /classify raises RuntimeError "expanded size of the
# tensor must match existing size" for inputs beyond ~2000 combined chars,
# then httpx sees the crashed worker as a ReadTimeout/RemoteProtocolError —
# and a crashed batch worker hangs every other request queued behind it,
# since Infinity batches multiple requests together, so this isn't just a
# per-row failure). Without this truncation, every long-context (premise,
# hypothesis) pair silently fell through EvalNLIClient.check()'s
# except-Exception branch and was scored as "neutral" — production's
# RAMService avoids this by bounding premise to MAX_PREMISE_CONTEXTS
# reranked chunks (see ram/service.py); eval scripts instead store the full
# raw retrieved_context blob (~15K chars on average, a concatenation of
# retrieved document chunks joined by a "\n\n---\n\n" separator), so naively
# truncating from the front discards the actual supporting chunk in the vast
# majority of rows (every row in Subset D exceeds the char budget). Instead
# we split on that separator and keep the chunk with the highest raw token
# overlap with the hypothesis sentence — a cheap, local stand-in for the
# reranker step RAMService performs in production. A generous char budget is
# used only to narrow down to the *right* chunk; the actual safety bound
# applied afterward is token-based (see EvalNLIClient._truncate_to_tokens) —
# a char-count heuristic isn't a reliable token-count proxy for dense
# Indonesian legal text (hyphenated document numbers, dates), which
# tokenizes far closer to 1 token/char than ordinary prose.
MAX_PREMISE_CHARS = 4000
def _select_relevant_chunk(premise: str, hypothesis: str, max_chars: int) -> str:
    """Pick the passage(s) of ``premise`` most likely to support ``hypothesis``.

    ``premise`` is a multi-chunk context joined with a plain blank line
    (``chat_service.py``'s context payload and ``RAMService.build_premise``
    both use bare ``"\\n\\n"`` — there is no distinguishing separator
    between chunks). The previous version of this function split on
    ``"\\n\\n-{3,}\\n\\n"``, a separator that is never actually produced
    anywhere in the codebase, so the split always no-op'd and every call
    fell through to a blind ``premise[:max_chars]`` — on Subset D's
    15k-22k-char contexts that means the NLI model only ever saw the
    first ~10% of the context, regardless of which part actually supports
    the hypothesis (see ``writing/overhaul.md`` M7/M22).

    This instead splits on blank lines (the real, if imperfect, boundary —
    chunk text can itself contain a blank line, so this is a heuristic,
    not an exact chunk recovery) and scores each paragraph by token
    containment of the hypothesis (``|hyp ∩ paragraph| / |hyp|``), then
    concatenates the highest-scoring paragraphs — in their original order,
    so multi-paragraph evidence still reads coherently — up to
    ``max_chars``.

    Args:
        premise: Full (possibly multi-chunk) retrieved context.
        hypothesis: Sentence being verified against the context.
        max_chars: Character budget for the returned excerpt.

    Returns:
        A char-budgeted excerpt of ``premise``, biased toward the
        paragraph(s) most relevant to ``hypothesis``.
    """
    if len(premise) <= max_chars:
        return premise

    paragraphs = [p for p in re.split(r"\n\s*\n", premise) if p.strip()]
    if len(paragraphs) <= 1:
        return premise[:max_chars]

    hyp_tokens = set(re.findall(r"\w+", hypothesis.lower()))
    if not hyp_tokens:
        return premise[:max_chars]

    def _containment(paragraph: str) -> float:
        para_tokens = set(re.findall(r"\w+", paragraph.lower()))
        if not para_tokens:
            return 0.0
        return len(hyp_tokens & para_tokens) / len(hyp_tokens)

    ranked = sorted(range(len(paragraphs)), key=lambda i: _containment(paragraphs[i]), reverse=True)

    selected: List[int] = []
    budget = max_chars
    for idx in ranked:
        if budget <= 0:
            break
        selected.append(idx)
        budget -= len(paragraphs[idx]) + 2  # +2 for the rejoining "\n\n"

    selected.sort()  # restore original document order
    excerpt = "\n\n".join(paragraphs[i] for i in selected)
    return excerpt[:max_chars]


class EvalNLIClient:
    """Minimal NLI adapter calling Infinity's /classify endpoint.

    Implements ``INLIModel`` for use with ``RAMService`` in eval scripts.

    Args:
        base_url: Infinity server base URL.
        model: NLI model identifier.
    """

    # Leaves margin under the model's real 514-position limit for whatever
    # special tokens Infinity's own tokenizer adds beyond ours.
    _MAX_TOTAL_TOKENS = 500
    _MAX_HYPOTHESIS_TOKENS = 150

    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        self._sep = " </s></s> " if "roberta" in model.lower() else " [SEP] "
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        self._tokenizer: Optional[Tokenizer] = None
        try:
            self._tokenizer = Tokenizer.from_pretrained(model)
            # tokenizer.json for this model ships its own truncation config
            # (max_length=128) that silently overrides any max_length we'd
            # pass to .encode() — disable it so our explicit token budgets
            # below (targeting the model's real 514-position limit, not
            # this arbitrary default) actually take effect.
            self._tokenizer.no_truncation()
        except Exception:
            pass

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or self._tokenizer is None:
            return text
        ids = self._tokenizer.encode(text).ids
        if len(ids) <= max_tokens:
            return text
        return self._tokenizer.decode(ids[:max_tokens])

    async def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI inference on a premise/hypothesis pair.

        Args:
            premise: Reference context.
            hypothesis: Statement to verify.

        Returns:
            NLIResult with canonical label and per-class scores.
        """
        selected_premise = _select_relevant_chunk(premise, hypothesis, MAX_PREMISE_CHARS)
        if self._tokenizer is not None:
            hypothesis = self._truncate_to_tokens(hypothesis, self._MAX_HYPOTHESIS_TOKENS)
            reserved = len(self._tokenizer.encode(hypothesis).ids) + len(self._tokenizer.encode(self._sep).ids)
            premise_budget = max(0, self._MAX_TOTAL_TOKENS - reserved)
            selected_premise = self._truncate_to_tokens(selected_premise, premise_budget)
        else:
            # Tokenizer unavailable (e.g. no network at startup) — fall back
            # to a conservative char cap as a last-resort safety net.
            hypothesis = hypothesis[:400]
            selected_premise = selected_premise[: max(0, 1000 - len(hypothesis) - len(self._sep))]

        text = f"{selected_premise}{self._sep}{hypothesis}"
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

    Requests are pinned to a single upstream provider, for the same reason the
    dataset-generation panel is (see ``_dataset_gen/panel.py``): one model slug
    can be served by several providers at different quantizations, so
    ``temperature=0.0`` alone does not make a result reproducible. A mid-run
    reroute silently changes the system under measurement — which matters most
    here, where the whole point of the Exp1a baseline row is to characterise one
    named model.

    Args:
        base_url: API base URL.
        api_key: API key.
        model: Default model identifier.
        provider_order: Comma-separated OpenRouter provider slugs to prefer, in
            order. Empty means no explicit preference, but fallbacks stay off.
        allow_fallbacks: Whether OpenRouter may reroute to another provider.
            Defaults to False so a run fails visibly rather than quietly
            measuring a different deployment.
        session_id: Accepted for call-site compatibility but **not sent**. It
            was introduced on the belief that it enabled provider-side prompt
            caching, and a cost claim rested on that. Measured directly
            (2026-07-22, gemini-2.5-flash, 6 identical calls per condition on a
            fresh prefix, the ``session_id`` condition run first so it could not
            inherit a warm cache): 3/6 cache hits with it, 2/6 without. Caching
            is automatic and prefix-driven. OpenRouter silently drops parameters
            it does not recognise, so sending it never errored and the belief
            went unchallenged. Reproduce with
            ``preflight.py --check-session-id``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider_order: str = "",
        allow_fallbacks: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        self._model = model
        self._provider_order = [p.strip() for p in provider_order.split(",") if p.strip()]
        self._allow_fallbacks = allow_fallbacks
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    @property
    def model(self) -> str:
        """The default model identifier this client was configured with."""
        return self._model

    def _provider_payload(self) -> Dict[str, Any]:
        """Build the OpenRouter provider-routing block."""
        provider: Dict[str, Any] = {"allow_fallbacks": self._allow_fallbacks}
        if self._provider_order:
            provider["order"] = self._provider_order
        return provider

    @staticmethod
    def _disable_qwen_thinking(model: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append Qwen's ``/no_think`` so reasoning models return a clean verdict.

        The same fix as ``LLMConnection._suppress_thinking`` but for the eval
        client, which calls OpenRouter directly: qwen3 ignores
        ``reasoning={"enabled": False}``, so without this the safety baseline
        and relevance judge get an empty or reasoning-contaminated response that
        ``parse_verdict`` classifies as INDETERMINATE for every row (measured:
        qwen3-14b/32b returned zero parseable verdicts -> empty predictions ->
        crash). Scoped to qwen so a stray control token never reaches deepseek etc.
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

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        session_id: Optional[str] = None,
    ) -> str:
        """Send a chat completion request and return the full response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Override model (defaults to self._model).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            session_id: Accepted and ignored (see class docstring).

        Returns:
            The assistant's response text.
        """
        target_model = model or self._model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": self._disable_qwen_thinking(target_model, messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
            "provider": self._provider_payload(),
        }
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        # Fall back to the reasoning field only if content is empty (some models
        # can't disable thinking); with /no_think on qwen, content is populated.
        return message.get("content") or message.get("reasoning") or ""

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 100,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion (implements ILLMJudgeConnection).

        Args:
            model: Model identifier.
            messages: List of message dicts.
            max_tokens: Maximum tokens to generate.
            session_id: Accepted and ignored (see class docstring).

        Yields:
            Text chunks as they arrive.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._disable_qwen_thinking(model, messages),
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
            "reasoning": {"enabled": False},
            "provider": self._provider_payload(),
        }
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload,
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


def get_llm_client_from_env(
    session_id: Optional[str] = None, model: Optional[str] = None
) -> EvalLLMClient:
    """Create an EvalLLMClient from environment variables.

    Reads:
        OPENROUTER_API_KEY: API key for OpenRouter.
        EVAL_LLM_MODEL: Model identifier (default: deepseek/deepseek-chat).
        EVAL_PROVIDER_ORDER: Comma-separated provider slugs to prefer.
        EVAL_ALLOW_FALLBACKS: "true" to permit rerouting to another provider.

    Args:
        session_id: Accepted and ignored (see EvalLLMClient docstring).
        model: Explicit model id override. When given, it wins over
            ``EVAL_LLM_MODEL`` — used by the production-faithful Exp1b judge
            pass to pin ``qwen/qwen3-14b`` (the production judge) instead of
            the eval default.

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
    model = model or os.environ.get("EVAL_LLM_MODEL", "deepseek/deepseek-chat")
    return EvalLLMClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        provider_order=os.environ.get("EVAL_PROVIDER_ORDER", ""),
        allow_fallbacks=os.environ.get("EVAL_ALLOW_FALLBACKS", "false").lower() == "true",
        session_id=session_id,
    )
