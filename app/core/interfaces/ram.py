"""
RAM-specific domain interfaces.
"""
from typing import List, Optional, Protocol, runtime_checkable

from app.core.interfaces.ai import EmbeddingResult, NLIResult
from app.core.interfaces.rag import RetrievedContext


@runtime_checkable
class IRAMService(Protocol):
    """Orchestrates the Response Assessment Module pipeline."""

    def build_premise(self, contexts: List[RetrievedContext]) -> str:
        """Concatenate KB parent chunk texts into a single NLI premise."""
        ...

    async def assess_sentence(
        self,
        sentence: str,
        premise: str,
        contexts: List[RetrievedContext],
        context_embs: Optional[List[EmbeddingResult]] = None,
    ) -> NLIResult:
        """Run NLI on a single sentence against the pre-built KB premise."""
        ...
