"""
Service layer for the Input Validation Module (IVM).

Responsible for checking prompts for malicious content and evaluating 
relevance scores of queries and documents against the knowledge base.
"""
from typing import List

import structlog

from .interfaces import IRelevanceStrategy, ISafetyModel

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
        relevance_strategy (IRelevanceStrategy): Strategy for evaluating relevance.
        similarity_threshold (float): Score threshold for semantic relevance.
    """

    def __init__(
        self,
        safety_model: ISafetyModel,
        relevance_strategy: IRelevanceStrategy,
        similarity_threshold: float,
    ):
        self.safety_model = safety_model
        self.relevance_strategy = relevance_strategy
        self.similarity_threshold = similarity_threshold

    async def check_malicious(self, query: str) -> None:
        """Validates the query against the safety model.

        Args:
            query (str): The prompt to check.

        Raises:
            MaliciousPromptException: If the prompt is malicious.
        """
        if not query.strip():
            return

        result = await self.safety_model.check_prompt(query)
        if not result.is_safe:
            logger.warning(
                "malicious_prompt_detected", 
                message=result.message,
                query=query
            )
            raise MaliciousPromptException("Malicious prompt detected.")

    def check_relevance(self, query: str, scores: List[float]) -> None:
        """Validates that the query is relevant based on a list of similarity scores.

        Args:
            query (str): The query string (used for logging).
            scores (List[float]): The similarity scores of the top-K retrieved chunks.

        Raises:
            IrrelevantQueryException: If the scores fail the relevance strategy.
        """
        if not query.strip():
            return
            
        is_relevant = self.relevance_strategy.evaluate(
            scores=scores, 
            similarity_threshold=self.similarity_threshold
        )
        
        logger.info(
            "relevance_check", 
            query=query, 
            scores=scores, 
            threshold=self.similarity_threshold,
            is_relevant=is_relevant
        )
        
        if not is_relevant:
            logger.warning(
                "irrelevant_query_detected", 
                scores=scores, 
                threshold=self.similarity_threshold
            )
            raise IrrelevantQueryException("Query is not relevant to the knowledge base.")

    def validate_document_relevance(self, document_chunk_scores: List[List[float]]) -> None:
        """Validates that an uploaded document is relevant to the knowledge base.

        Args:
            document_chunk_scores (List[List[float]]): A list of score lists. 
                Each inner list represents the top-K search scores for one sampled chunk of the document.

        Raises:
            IrrelevantDocumentException: If all sampled chunks fall below similarity_threshold.
        """
        if not document_chunk_scores:
            return

        best_overall_score = 0.0

        for scores in document_chunk_scores:
            is_relevant = self.relevance_strategy.evaluate(
                scores=scores,
                similarity_threshold=self.similarity_threshold
            )

            if is_relevant:
                # Found at least one relevant chunk, document is allowed
                logger.info(
                    "document_relevant", 
                    scores=scores, 
                    threshold=self.similarity_threshold
                )
                return

            if scores:
                best_overall_score = max(best_overall_score, max(scores))

        # If we get here, none of the sampled chunks were relevant
        logger.warning(
            "irrelevant_document_detected", 
            max_score=best_overall_score, 
            threshold=self.similarity_threshold
        )
        raise IrrelevantDocumentException("Uploaded document is not relevant to the knowledge base.")
