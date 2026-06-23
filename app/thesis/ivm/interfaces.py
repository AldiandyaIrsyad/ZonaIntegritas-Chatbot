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


class IRelevanceStrategy(Protocol):
    """Strategy for evaluating query relevance against a list of similarity scores."""
    
    def evaluate(self, scores: List[float], similarity_threshold: float) -> bool:
        """Evaluates relevance based purely on a list of floating point scores.
        
        Args:
            scores: List of similarity scores (e.g. cosine similarities).
            similarity_threshold: The configured relevance threshold.
            
        Returns:
            bool: True if relevant, False if irrelevant/flagged.
        """
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