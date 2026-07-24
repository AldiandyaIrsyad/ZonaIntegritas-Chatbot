"""Dense + sparse text embedding adapter over the Infinity HTTP server.

Fulfills: ``app/kb/domain/interfaces.py::ITextEmbedder``.
Not currently wired into ``app/kb/dependency.py`` — production ingestion
and search use ``bge_m3_embeddings.py::BGEM3Embeddings`` instead, because
Infinity's own model list registers ``BAAI/bge-m3`` as dense-only (see
that module's docstring and ``docs/02-arsitektur.md`` §2.1). This adapter
is kept as the HTTP-backed alternative implementation the interface
docstring references, and remains usable for any Infinity-hosted embedding
model that does support sparse output.
"""

import httpx
import structlog
from typing import List

from app.kb.domain.interfaces import ITextEmbedder, EmbeddingResult

logger = structlog.get_logger(__name__)

class InfinityEmbeddings(ITextEmbedder):
    """HTTP adapter for the Infinity embedding server.

    Fulfills: ``app/kb/domain/interfaces.py::ITextEmbedder``.
    """

    def __init__(self, base_url: str, model: str, batch_size: int = 8) -> None:
        """Open an HTTP client for the Infinity server.

        Args:
            base_url: Infinity server base URL.
            model: Model identifier registered with Infinity.
            batch_size: Max texts sent per ``/embeddings`` request.
        """
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info("InfinityEmbeddings initialized", model=model, base_url=base_url)

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed texts in chunks of ``batch_size``, requesting sparse output
        alongside the dense vectors from Infinity's ``/embeddings`` endpoint."""
        if not texts:
            return []

        all_results: List[EmbeddingResult] = []

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
            except Exception as exc:
                logger.error("embed.failed", batch_index=i, error=str(exc))
                raise

            embeddings_data = sorted(
                response_data.get("data", []),
                key=lambda x: x["index"],
            )

            for data_obj in embeddings_data:
                dense_vec: List[float] = data_obj.get("embedding", [])
                sparse_raw: dict[str, float] = data_obj.get("sparse_embedding") or {}
                sparse_indices: List[int] = []
                sparse_values: List[float] = []
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

        return all_results

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()
