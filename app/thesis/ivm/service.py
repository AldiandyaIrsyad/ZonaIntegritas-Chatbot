"""
Service layer for the Input Validation Module (IVM).

Responsible for checking prompts for malicious content and evaluating 
relevance of queries and documents against the knowledge base.
"""
from typing import List

import structlog

from .interfaces import ISafetyModel, IJudge

logger = structlog.get_logger(__name__)


class IVMException(Exception):
    """Base exception for IVM validation failures."""
    pass


class MaliciousPromptException(IVMException):
    """Raised when a prompt fails the safety check."""
    pass


class IrrelevantQueryException(IVMException):
    """Raised when a query is deemed irrelevant to the knowledge base."""
    pass


class IrrelevantDocumentException(IVMException):
    """Raised when a document is deemed irrelevant to the knowledge base."""
    pass


class IVMService:
    """Input Validation Module (IVM)
    
    Responsible for validating prompts and relevance scores.
    
    Args:
        safety_model (ISafetyModel): Client for prompt injection checks.
        judge (IJudge): Strategy for evaluating relevance using an LLM.
    """

    def __init__(
        self,
        safety_model: ISafetyModel,
        judge: IJudge,
    ):
        self.safety_model = safety_model
        self.judge = judge

    async def check_malicious(self, query: str) -> None:
        """Validates the query against the safety model using a sliding window.

        Args:
            query (str): The prompt to check.

        Raises:
            MaliciousPromptException: If the prompt is malicious or if the service fails.
        """
        if not query.strip():
            return

        window_size = 512
        overlap = 50
        
        # Sliding window
        start = 0
        while start < len(query) or start == 0: # Ensures it runs at least once even if query is short
            end = start + window_size
            chunk = query[start:end]
            
            try:
                result = await self.safety_model.check_prompt(chunk)
            except Exception as e:
                logger.error("safety_model.error", error=str(e), exc_info=True)
                raise MaliciousPromptException("Safety check failed due to internal error.") from e
                
            if not result.is_safe:
                logger.warning(
                    "malicious_prompt_detected", 
                    message=result.message,
                    chunk_preview=chunk[:50]
                )
                raise MaliciousPromptException("Malicious prompt detected.")
                
            if end >= len(query):
                break
            start += window_size - overlap

    async def check_relevance(self, query: str, context_chunks: List[str]) -> None:
        """Validates that the query is relevant to the retrieved contexts using LLM Judge.

        Args:
            query (str): The query string.
            context_chunks (List[str]): Text chunks retrieved from the knowledge base.

        Raises:
            IrrelevantQueryException: If the query is deemed irrelevant or the service fails.
        """
        if not query.strip() or not context_chunks:
            raise IrrelevantQueryException("Query or contexts are empty.")
            
        combined_context = "\n".join(context_chunks)
        
        try:
            is_relevant = await self.judge.evaluate_relevance(query, combined_context)
        except Exception as e:
            logger.error("judge.evaluate_relevance.error", error=str(e), exc_info=True)
            raise IrrelevantQueryException("Relevance check failed due to internal error.") from e
        
        logger.info(
            "relevance_check", 
            query=query, 
            is_relevant=is_relevant
        )
        
        if not is_relevant:
            logger.warning(
                "irrelevant_query_detected", 
                query=query
            )
            raise IrrelevantQueryException("Query is not relevant to the knowledge base.")

    async def validate_document_relevance(self, document_chunks: List[str]) -> None:
        """Validates that an uploaded document is relevant to the domain.

        Args:
            document_chunks (List[str]): Sampled text chunks from the document.

        Raises:
            IrrelevantDocumentException: If the document chunks are deemed irrelevant.
        """
        if not document_chunks:
            raise IrrelevantDocumentException("Document is empty.")

        domain_query = "Is this document relevant to the knowledge base domain and safe to ingest?"
        
        for chunk in document_chunks:
            try:
                is_relevant = await self.judge.evaluate_relevance(domain_query, chunk)
            except Exception as e:
                logger.error("judge.validate_document.error", error=str(e), exc_info=True)
                raise IrrelevantDocumentException("Document relevance check failed.") from e
                
            if is_relevant:
                # Found at least one relevant chunk, allow it.
                logger.info("document_relevant", chunk_preview=chunk[:50])
                return

        # None of the chunks were relevant
        logger.warning("irrelevant_document_detected")
        raise IrrelevantDocumentException("Uploaded document is not relevant to the knowledge base.")
