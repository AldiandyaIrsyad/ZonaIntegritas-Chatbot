"""IVM domain interfaces."""
from typing import List, Protocol

from app.core.interfaces.ai import EmbeddingResult
from app.core.interfaces.infra import SearchResult


class IRelevanceStrategy(Protocol):
    """Strategy for evaluating query relevance against vector search results."""
    
    def evaluate(self, results: List[SearchResult], similarity_threshold: float) -> bool:
        """Evaluates relevance of the results.
        
        Args:
            results: List of SearchResult from the vector store.
            similarity_threshold: The configured relevance threshold.
            
        Returns:
            bool: True if relevant, False if irrelevant/flagged.
        """
        ...


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
