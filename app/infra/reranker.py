"""Cross-encoder reranking adapter.

Wraps HTTP calls to the Infinity ``/rerank`` endpoint.  Sends ``(query,
documents)`` pairs to a cross-encoder model and returns documents sorted by
descending relevance score.

The Infinity server handles model lifecycle and batching; this client is a
thin, stateless async HTTP wrapper.
"""

from __future__ import annotations

import httpx
import structlog

from app.core.interfaces.ai import IReranker, RankedResult

logger = structlog.get_logger(__name__)


class Reranker:
    """HTTP adapter for the Infinity cross-encoder reranking server.

    Sends query + document pairs to ``/rerank`` and returns the results ranked
    by descending relevance score.  Satisfies the
    :class:`~app.core.interfaces.ai.IReranker` Protocol structurally.

    Args:
        base_url: Base URL of the Infinity server
                  (e.g. ``"http://infinity:7997"``).
        model: Reranker model identifier
               (e.g. ``"BAAI/bge-reranker-v2-m3"``).
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        logger.info(
            "Reranker initialised",
            model=model,
            base_url=base_url,
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 3,
    ) -> list[RankedResult]:
        """Score and rank candidate documents by relevance to the query.

        Returns an empty list (rather than raising) on HTTP or network errors
        so that the calling pipeline can degrade gracefully.  Invalid result
        items (bad index or non-numeric score) are skipped with a warning.

        Args:
            query: The user's search query.
            documents: Candidate document texts to score.
            top_k: Maximum number of results to return.  Returns ``[]``
                   immediately if ``<= 0``.

        Returns:
            Up to ``top_k`` :class:`~app.core.interfaces.ai.RankedResult`
            items sorted by descending relevance score.
        """
        if not documents or top_k <= 0:
            return []

        try:
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
        except httpx.HTTPStatusError as exc:
            logger.error(
                "rerank.http_error",
                status_code=exc.response.status_code,
                error=str(exc),
            )
            return []
        except Exception as exc:
            logger.error("rerank.request_failed", error=str(exc))
            return []

        results: list[RankedResult] = []
        for item in data.get("results", []):
            idx = item.get("index")
            score = item.get("relevance_score")

            if idx is None or score is None:
                continue
            if not isinstance(idx, int) or idx < 0 or idx >= len(documents):
                logger.warning(
                    "rerank.invalid_index",
                    item=item,
                    document_count=len(documents),
                )
                continue
            if not isinstance(score, (int, float)):
                logger.warning("rerank.invalid_score", item=item)
                continue

            results.append(
                RankedResult(
                    index=idx,
                    text=documents[idx],
                    score=float(score),
                )
            )

        # Sort defensively — the server usually returns sorted results
        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_k]

        logger.debug(
            "rerank.complete",
            document_count=len(documents),
            result_count=len(top),
            top_k=top_k,
        )
        return top

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
