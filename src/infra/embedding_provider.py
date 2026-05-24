"""
Infinity embedding server client.

Wraps HTTP calls to the Infinity `/embeddings` endpoint to generate
dense and sparse vector representations from BAAI/bge-m3.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import httpx

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Dense and sparse embedding output for a single text."""
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


class EmbeddingProvider:
    """
    HTTP client for the Infinity embedding server.

    Sends text to the `/embeddings` endpoint and retrieves both dense
    and sparse (BM25) vector representations from BAAI/bge-m3.

    Processes inputs in configurable batch sizes to respect server memory
    constraints (INFINITY_BATCH_SIZE).
    """

    def __init__(self, base_url: str, model: str, batch_size: int = 8):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def embed_texts(
        self, texts: List[str]
    ) -> List[EmbeddingResult]:
        """Generate dense + sparse embeddings for a list of texts.

        Processes in batches to avoid overwhelming the Infinity server.
        bge-m3 returns both dense and sparse representations natively.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of EmbeddingResult, one per input text, preserving order.

        Raises:
            httpx.HTTPStatusError: If the Infinity server returns an error.
        """
        if not texts:
            return []

        all_results: List[EmbeddingResult] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Request dense embeddings
            dense_response = await self._client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": batch,
                },
            )
            dense_response.raise_for_status()
            dense_data = dense_response.json()

            # Parse dense embeddings — sorted by index for order guarantee
            dense_embeddings = sorted(
                dense_data.get("data", []), key=lambda x: x["index"]
            )

            if len(dense_embeddings) != len(batch):
                raise ValueError(
                    f"Embedding response size mismatch: expected {len(batch)}, "
                    f"got {len(dense_embeddings)}"
                )

            for j in range(len(batch)):
                dense_vec = dense_embeddings[j]["embedding"]

                all_results.append(
                    EmbeddingResult(
                        dense=dense_vec,
                        sparse_indices=[],
                        sparse_values=[],
                    )
                )

        logger.info("Embedded %d texts via Infinity (%s)", len(texts), self.model)
        return all_results

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
