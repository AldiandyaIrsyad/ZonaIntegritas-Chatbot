"""
Relevance / out-of-domain (OOD) checking service for the IVM.

Delegates the actual relevance decision to an injected IRelevanceChecker
(see app/thesis/ivm/checkers.py), wired at app/chat/dependency.py.
"""
from typing import List, Optional

import structlog

from .interfaces import IRelevanceChecker

logger = structlog.get_logger(__name__)


class RelevanceException(Exception):
    """Base exception for relevance/OOD validation failures."""
    pass


class IrrelevantQueryException(RelevanceException):
    """Raised when a query is deemed irrelevant to the knowledge base."""
    pass


class RelevanceService:
    """Validates that queries and documents are in-domain for the knowledge base.

    Args:
        relevance_checker (IRelevanceChecker): The active OOD-check backend.
    """

    def __init__(self, relevance_checker: IRelevanceChecker):
        self.relevance_checker = relevance_checker

    async def check_relevance(
        self,
        query: str,
        context_chunks: List[str],
        context_scores: Optional[List[float]] = None,
    ) -> None:
        """Validates that the query is relevant to the retrieved contexts.

        Args:
            query (str): The query string.
            context_chunks (List[str]): Text chunks retrieved from the knowledge base.
            context_scores (Optional[List[float]]): Similarity scores for
                ``context_chunks`` (same order), already computed by
                retrieval. Passed through so RAG-only checkers can skip a
                redundant embedding/search call.

        Raises:
            IrrelevantQueryException: If the query is deemed irrelevant or the service fails.
        """
        if not query.strip() or not context_chunks:
            raise IrrelevantQueryException("Query or contexts are empty.")

        try:
            is_relevant = await self.relevance_checker.check_query(
                query, context_chunks, context_scores or []
            )
        except Exception as e:
            logger.error("relevance_checker.check_query.error", error=str(e), exc_info=True)
            raise IrrelevantQueryException("Relevance check failed due to internal error.") from e

        logger.info("relevance_check", query=query, is_relevant=is_relevant)

        if not is_relevant:
            logger.warning("irrelevant_query_detected", query=query)
            raise IrrelevantQueryException("Query is not relevant to the knowledge base.")
