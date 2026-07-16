"""
Concrete IRelevanceChecker implementations.

- LLMJudgeRelevanceChecker: wraps the existing LLM-as-judge (IJudge), one
  call per check, fails closed on error.
"""
from typing import List

import structlog

from .interfaces import IJudge, IRelevanceChecker

logger = structlog.get_logger(__name__)


class LLMJudgeRelevanceChecker(IRelevanceChecker):
    """Relevance backend using an LLM-as-judge (today's default method)."""

    def __init__(self, judge: IJudge):
        self.judge = judge

    async def check_query(
        self, query: str, context_chunks: List[str], context_scores: List[float]
    ) -> bool:
        combined_context = "\n".join(context_chunks)
        return await self.judge.evaluate_relevance(query, combined_context)
