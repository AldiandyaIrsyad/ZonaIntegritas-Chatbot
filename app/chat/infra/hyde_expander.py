"""HyDE (Hypothetical Document Embeddings) query expander.

This adapter implements :class:`app.kb.domain.interfaces.IQueryExpander`
using an :class:`app.chat.domain.interfaces.ILLMConnection` to generate a
hypothetical answer document for a given user query. The generated document
is then embedded (instead of the raw query) to improve retrieval recall.

Dependency rule compliance:
    - ``chat/infra`` may import ``chat/domain`` and ``kb/domain`` (Protocols).
    - It does NOT import ``kb/application`` or ``kb/infra``.
"""

from typing import Dict, List

import structlog

from app.chat.domain.interfaces import ILLMConnection
from app.kb.domain.interfaces import IQueryExpander

logger = structlog.get_logger(__name__)


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
    ) -> None:
        """Initialize the HyDE expander.

        Args:
            llm: An LLM connection supporting non-streaming ``generate()``.
            model: Model identifier to use for generation.
            prompt_template: Prompt template containing ``{query}`` placeholder.
            system_prompt: System message setting the domain/register the
                generated hypothetical document should imitate.
            max_tokens: Max tokens for the hypothetical document.
            temperature: Sampling temperature (0.0 = deterministic).
        """
        self._llm = llm
        self._model = model
        self._prompt_template = prompt_template
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def expand(self, query: str) -> str:
        """Generate a hypothetical answer document for the query.

        Args:
            query: The user's raw search query.

        Returns:
            A hypothetical answer paragraph to be embedded for retrieval.
        """
        if not query.strip():
            return ""

        user_prompt = self._prompt_template.replace("{query}", query)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
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
