"""
Infinity reranker client.

Wraps HTTP calls to the Infinity `/rerank` endpoint to score and sort
retrieved document chunks by query relevance using a cross-encoder model.
"""
import logging
from dataclasses import dataclass
from typing import List

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A document scored by the reranker."""
    index: int
    text: str
    score: float


class Reranker:
    """
    HTTP client for the Infinity cross-encoder reranking server.

    Sends query + document pairs to the `/rerank` endpoint and returns
    documents sorted by descending relevance score.

    Args:
        base_url (str): The base URL of the Infinity server.
        model (str): The reranker model ID to use.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
    ) -> List[RankedResult]:
        """Rerank documents against a query using the cross-encoder model.

        Args:
            query (str): The user's search query.
            documents (List[str]): List of document texts to score.
            top_k (int, optional): Number of top results to return. Defaults to 3.

        Returns:
            List[RankedResult]: List of RankedResult sorted by descending score, truncated to top_k.

        Raises:
            httpx.HTTPStatusError: If the Infinity server returns an error.
        """
        if not documents or top_k <= 0:
            return []

        response = await self._client.post(
            "/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            idx = item.get("index")
            score = item.get("relevance_score")
            if idx is None or score is None:
                continue

            if not isinstance(idx, int) or idx < 0 or idx >= len(documents):
                logger.warning("Skipping rerank result with invalid index: %r", item)
                continue
            
            if not isinstance(score, (int, float)):
                logger.warning("Skipping rerank result with invalid score: %r", item)
                continue

            results.append(
                RankedResult(
                    index=idx,
                    text=documents[idx],
                    score=float(score),
                )
            )

        # Sort by score descending (server usually does this, but be safe)
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Reranked %d documents, returning top %d",
            len(documents),
            min(top_k, len(results)),
        )
        return results[:top_k]

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
