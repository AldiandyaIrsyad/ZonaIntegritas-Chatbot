"""HyDE (Hypothetical Document Embeddings) query expander.

This adapter implements :class:`app.kb.domain.interfaces.IQueryExpander`
using an :class:`app.chat.domain.interfaces.ILLMConnection` to generate a
hypothetical answer document for a given user query. The generated document
is then embedded (instead of the raw query) to improve retrieval recall.

Dependency rule compliance:
    - ``chat/infra`` may import ``chat/domain`` and ``kb/domain`` (Protocols).
    - It does NOT import ``kb/application`` or ``kb/infra``.
"""

import asyncio
import time
from typing import Dict, List, Optional

import structlog

from app.chat.domain.interfaces import ILLMConnection
from app.kb.domain.interfaces import IKBRepository, IQueryExpander

logger = structlog.get_logger(__name__)

# Module-level (not per-instance) TTL cache for the KB grounding context —
# ``get_query_expander`` builds a fresh HyDEExpander per request, so an
# instance-level cache would never survive between requests.
_kb_context_cache: Dict[str, object] = {"text": "", "expires_at": 0.0}
_kb_context_lock = asyncio.Lock()


class HyDEExpander(IQueryExpander):
    """Generates a hypothetical document for HyDE retrieval.

    Uses an LLM to produce a short answer-like paragraph for the user's
    query. The paragraph is semantically closer to the knowledge-base
    documents than the question phrasing, so embedding it yields better
    vector search recall.
    """

    def __init__(
        self,
        llm: ILLMConnection,
        model: str,
        prompt_template: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        kb_repo: Optional[IKBRepository] = None,
        context_max_docs: int = 20,
        context_refresh_seconds: int = 300,
    ) -> None:
        """Initialize the HyDE expander.

        Args:
            llm: An LLM connection supporting non-streaming ``generate()``.
            model: Model identifier to use for generation.
            prompt_template: Prompt template containing ``{query}`` placeholder.
            system_prompt: System message setting the domain/register the
                generated hypothetical document should imitate. May contain
                a ``{kb_context}`` placeholder (see ``kb_repo``).
            max_tokens: Max tokens for the hypothetical document.
            temperature: Sampling temperature (0.0 = deterministic).
            kb_repo: Optional KB repository used to ground the prompt with
                the actual document titles/descriptions in this KB (fills
                the ``{kb_context}`` placeholder). If ``None``, the
                placeholder resolves to an empty string.
            context_max_docs: Max number of active documents to list in the
                grounding context.
            context_refresh_seconds: TTL for the module-level grounding
                context cache.
        """
        self._llm = llm
        self._model = model
        self._prompt_template = prompt_template
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._kb_repo = kb_repo
        self._context_max_docs = context_max_docs
        self._context_refresh_seconds = context_refresh_seconds

    async def _get_kb_context(self) -> str:
        """Return a cached, TTL-refreshed grounding block of active KB docs."""
        if self._kb_repo is None:
            return ""

        if time.monotonic() < _kb_context_cache["expires_at"]:
            return _kb_context_cache["text"]

        async with _kb_context_lock:
            # Re-check after acquiring the lock — another request may have
            # already refreshed it while we were waiting.
            if time.monotonic() < _kb_context_cache["expires_at"]:
                return _kb_context_cache["text"]

            docs = await self._kb_repo.get_all_pdfs()
            active_docs = [d for d in docs if d.active][: self._context_max_docs]
            lines = [
                f"- {doc.title}: {(doc.description or '')[:150]}"
                for doc in active_docs
            ]
            text = "\n".join(lines)

            _kb_context_cache["text"] = text
            _kb_context_cache["expires_at"] = time.monotonic() + self._context_refresh_seconds
            return text

    async def expand(self, query: str) -> str:
        """Generate a hypothetical answer document for the query.

        Args:
            query: The user's raw search query.

        Returns:
            A hypothetical answer paragraph to be embedded for retrieval.
        """
        if not query.strip():
            return ""

        kb_context = await self._get_kb_context()
        system_prompt = self._system_prompt.replace("{kb_context}", kb_context)
        user_prompt = self._prompt_template.replace("{query}", query).replace("{kb_context}", kb_context)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug(
            "chat.hyde.expand_start",
            model=self._model,
            query_len=len(query),
            max_tokens=self._max_tokens,
        )

        try:
            doc = await self._llm.generate(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            logger.debug(
                "chat.hyde.expand_done",
                hyde_len=len(doc),
            )
            return doc.strip()
        except Exception as exc:
            logger.error("chat.hyde.expand_error", error=str(exc), exc_info=True)
            raise

    async def close(self) -> None:
        """No-op — the LLM connection lifecycle is managed by the caller."""
        pass
