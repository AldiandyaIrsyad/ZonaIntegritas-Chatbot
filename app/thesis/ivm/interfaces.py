"""Ports (Protocol interfaces) for the Input Validation Module (IVM).

The IVM is part of the pure ``thesis`` research core and defines its own
ports so it never imports the chat/kb infra layers directly (Dependency
Inversion). Concrete adapters live in ``app/thesis/ivm/`` and
``app/chat/infra/`` and are wired in ``app/chat/dependency.py``.

Ports → adapters map:
    - :class:`ISafetyModel`           → ``app/chat/infra/prompt_guard_client.py::PromptGuardClient``
                                        (wired in ``app/chat/dependency.py::get_prompt_guard_client``)
    - :class:`ILLMJudgeConnection`    → ``app/chat/infra/llm_connection.py::LLMConnection``
                                        (the same adapter fulfills the chat
                                        ``ILLMConnection`` port; this is a
                                        narrower view for the judge)
    - :class:`IJudge`                 → ``app/thesis/ivm/judge.py::LLMJudge``
    - :class:`IRelevanceChecker`      → ``app/thesis/ivm/checkers.py``:
                                        ``LLMJudgeRelevanceChecker`` (default),
                                        ``SimilarityThresholdRelevanceChecker``,
                                        ``NliEntailmentRelevanceChecker``
"""

from dataclasses import dataclass
from typing import List, Protocol, AsyncIterator, Any, Dict


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a single prompt injection classification request.

    Attributes:
        is_safe: True if the input passed the safety classifier.
        message: Human-readable reason / raw classifier output.
    """

    is_safe: bool
    message: str


class ISafetyModel(Protocol):
    """Port for prompt injection detection adapters in the research core.

    Implemented by: ``app/chat/infra/prompt_guard_client.py::PromptGuardClient``
    (an Infinity-hosted Llama-Prompt-Guard-2-86M client; wired in
    ``app/chat/dependency.py::get_prompt_guard_client``).
    """

    async def check_prompt(self, text: str) -> SafetyResult:
        """Classify a user input for prompt injection or jailbreak attempts.

        Args:
            text: The user input to classify.

        Returns:
            SafetyResult with the verdict and reason.
        """
        ...


class ILLMJudgeConnection(Protocol):
    """Port for an LLM connection used by the relevance judge.

    This is a deliberately narrower view than the chat module's
    :class:`ILLMConnection` — it exposes only ``stream_chat`` — so the
    judge depends on the minimal surface it needs. This breaks the
    dependency on the chat module's ``ILLMConnection``.

    Implemented by: ``app/chat/infra/llm_connection.py::LLMConnection``
    (the same adapter fulfills both ports; wired in
    ``app/chat/dependency.py::get_llm_connection``).
    """

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 100,
    ) -> AsyncIterator[str]:
        """Stream a chat completion.

        Args:
            model: Model identifier.
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.

        Yields:
            Incremental response text fragments.
        """
        ...


class IJudge(Protocol):
    """Port for an LLM-based relevance judge.

    Implemented by: ``app/thesis/ivm/judge.py::LLMJudge``
    (wired in ``app/chat/dependency.py::get_relevance_service``).
    """

    async def evaluate_relevance(self, query: str, context: str) -> bool:
        """Evaluate if the given query is relevant to the provided context.

        Args:
            query: The user query to evaluate.
            context: The text context to check relevance against.

        Returns:
            bool: True if the query is relevant to the context, False otherwise.
        """
        ...


class IRelevanceChecker(Protocol):
    """Port for an OOD/relevance decision backend.

    Used by ``RelevanceService`` to decide whether a user query is in-domain.
    Implementations live in ``app/thesis/ivm/checkers.py``:
        - ``LLMJudgeRelevanceChecker`` (default) — wraps :class:`IJudge`.
        - ``SimilarityThresholdRelevanceChecker`` — kNN-OOD framing, no model call.
        - ``NliEntailmentRelevanceChecker`` — thresholds NLI entailment score.
    """

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        """Decide whether a user query is in-domain.

        Args:
            query: The user's raw query text.
            context_chunks: Text of the top retrieved KB contexts for this query.
            context_scores: Similarity scores (same order) already computed
                by retrieval, so implementations that only need scores can
                skip re-embedding/re-searching.

        Returns:
            bool: True if the query is relevant/in-domain.
        """
        ...