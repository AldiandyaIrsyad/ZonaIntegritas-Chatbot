"""IVM domain interfaces."""
from typing import List, Protocol

from app.core.interfaces.ai import EmbeddingResult


class IIVMService(Protocol):
    """Input Validation Module (IVM) service contract."""

    async def validate_prompt(self, query: str) -> None:
        """Validates the user's prompt for injection and relevance.
        
        Args:
            query (str): The raw text of the user's prompt.
            
        Raises:
            HTTPException: If the prompt is malicious or irrelevant.
        """
        ...

    async def validate_document_relevance(self, embeddings: List[EmbeddingResult]) -> None:
        """Validates that an uploaded document is relevant to the knowledge base.
        
        Args:
            embeddings (list[EmbeddingResult]): A list of EmbeddingResult objects for the document.
            
        Raises:
            HTTPException: If all sampled chunks fall below similarity_threshold.
        """
        ...
