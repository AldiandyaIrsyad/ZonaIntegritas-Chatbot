from dataclasses import dataclass
from typing import List, Protocol, AsyncIterator, Any, Dict


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a single prompt injection classification request."""
    is_safe: bool
    message: str


class ISafetyModel(Protocol):
    """Structural contract for prompt injection detection adapters in the research core."""

    async def check_prompt(self, text: str) -> SafetyResult:
        """Classify a user input for prompt injection or jailbreak attempts."""
        ...


class ILLMJudgeConnection(Protocol):
    """Structural contract for an LLM connection used by the judge.
    
    This breaks the dependency on the chat module's ILLMConnection.
    """

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 100,
    ) -> AsyncIterator[str]:
        """Stream a chat completion."""
        ...


class IJudge(Protocol):
    """Structural contract for an LLM-based judge."""

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
    """Structural contract for an OOD/relevance decision backend.

    Used by ``RelevanceService`` to decide whether a user query is in-domain.
    Implementations live in ``app/thesis/ivm/checkers.py``.
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