"""Dense + sparse text embedding adapter.

Wraps HTTP calls to the Infinity ``/embeddings`` endpoint to produce dense
and BM25-sparse vector representations using BAAI/bge-m3.  Inputs are
processed in configurable batches to respect server memory constraints.

The Infinity server handles model lifecycle and GPU scheduling; this client is
a thin, stateless async HTTP wrapper.
"""

from __future__ import annotations

import httpx
import structlog

from app.core.interfaces.ai import EmbeddingResult, IEmbeddingProvider

logger = structlog.get_logger(__name__)


class EmbeddingProvider:
    """HTTP adapter for the Infinity embedding server.

    Retrieves both dense and BM25-sparse vector representations for a list of
    texts.  Inputs are split into batches of :attr:`batch_size` to avoid
    overwhelming the server.  Satisfies the
    :class:`~app.core.interfaces.ai.IEmbeddingProvider` Protocol structurally.

    Args:
        base_url: Base URL of the Infinity server
                  (e.g. ``"http://infinity:7997"``).
        model: Embedding model identifier (e.g. ``"BAAI/bge-m3"``).
        batch_size: Number of texts to send per HTTP request.  Defaults to
                    ``8`` (tune via ``INFINITY_BATCH_SIZE``).
    """

    def __init__(self, base_url: str, model: str, batch_size: int = 8) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info(
            "EmbeddingProvider initialised",
            model=model,
            base_url=base_url,
            batch_size=batch_size,
        )

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate dense + sparse embeddings for a list of texts.

        Processes texts in batches of :attr:`batch_size`.  Output order is
        guaranteed to match input order via the ``index`` field returned by the
        Infinity server.

        Args:
            texts: Input strings to embed.

        Returns:
            One :class:`~app.core.interfaces.ai.EmbeddingResult` per input
            text, in the same order.

        Raises:
            ValueError: If the server returns a different number of embeddings
                than were sent in a batch (indicates a server-side bug).
            httpx.HTTPStatusError: Re-raised if the Infinity server returns a
                non-2xx response.
            Exception: Re-raised for any other network or deserialization error.
        """
        if not texts:
            return []

        all_results: list[EmbeddingResult] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = await self._client.post(
                    "/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                        "return_sparse": True,
                    },
                )
                response.raise_for_status()
                response_data = response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "embed.http_error",
                    batch_index=i,
                    batch_size=len(batch),
                    status_code=exc.response.status_code,
                    error=str(exc),
                )
                raise
            except Exception as exc:
                logger.error(
                    "embed.request_failed",
                    batch_index=i,
                    batch_size=len(batch),
                    error=str(exc),
                )
                raise

            # Sort by index to guarantee order — Infinity may reorder responses
            embeddings_data = sorted(
                response_data.get("data", []),
                key=lambda x: x["index"],
            )

            if len(embeddings_data) != len(batch):
                raise ValueError(
                    f"Embedding response size mismatch: expected {len(batch)}, "
                    f"got {len(embeddings_data)} (batch starting at index {i})"
                )

            for data_obj in embeddings_data:
                dense_vec: list[float] = data_obj.get("embedding", [])
                sparse_raw: dict[str, float] = data_obj.get("sparse_embedding") or {}
                sparse_indices: list[int] = []
                sparse_values: list[float] = []
                for k, v in sparse_raw.items():
                    sparse_indices.append(int(k))
                    sparse_values.append(float(v))

                all_results.append(
                    EmbeddingResult(
                        dense=dense_vec,
                        sparse_indices=sparse_indices,
                        sparse_values=sparse_values,
                    )
                )

        logger.debug(
            "embed.complete",
            text_count=len(texts),
            model=self.model,
        )
        return all_results

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
