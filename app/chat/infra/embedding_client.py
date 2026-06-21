"""Embedding client for the Chat module (used by RAM)."""

import httpx
import structlog
from typing import List

from app.thesis.ram.interfaces import IEmbeddingModel

logger = structlog.get_logger(__name__)

class EmbeddingClient(IEmbeddingModel):
    """Infrastructure adapter for generating dense embeddings via Infinity HTTP."""

    def __init__(self, base_url: str, model: str, batch_size: int = 8) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=5.0),
        )

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_results: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = await self._client.post(
                    "/embeddings",
                    json={
                        "model": self.model,
                        "input": batch,
                    },
                )
                response.raise_for_status()
                response_data = response.json()
            except Exception as exc:
                logger.error("chat.embed.failed", batch_index=i, error=str(exc))
                raise

            embeddings_data = sorted(
                response_data.get("data", []),
                key=lambda x: x["index"],
            )

            for data_obj in embeddings_data:
                dense_vec: List[float] = data_obj.get("embedding", [])
                all_results.append(dense_vec)

        return all_results

    async def close(self) -> None:
        await self._client.aclose()
