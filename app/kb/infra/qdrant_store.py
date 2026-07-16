"""Qdrant vector database adapter."""

from typing import Any, Optional, List
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType,
    PointStruct, SparseVector, SparseVectorParams, VectorParams
)
from qdrant_client import models

from app.kb.domain.interfaces import IVectorStore, ChunkVector, SearchResult

logger = structlog.get_logger(__name__)

BGE_M3_DENSE_DIM: int = 1024

class QdrantStore(IVectorStore):
    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self.collection_name = collection_name
        self._client = AsyncQdrantClient(host=host, port=port)
        logger.info("QdrantStore initialized", host=host, port=port, collection_name=collection_name)

    async def ensure_collection(self) -> None:
        try:
            collections = await self._client.get_collections()
            existing = {c.name for c in collections.collections}

            if self.collection_name in existing:
                return

            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=BGE_M3_DENSE_DIM, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

            for field_name, field_schema in [
                ("is_active", PayloadSchemaType.BOOL),
                ("session_id", PayloadSchemaType.KEYWORD),
                ("doc_id", PayloadSchemaType.KEYWORD),
                ("content_type", PayloadSchemaType.KEYWORD),
            ]:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

        except Exception as exc:
            logger.error("qdrant.ensure_failed", error=str(exc))
            raise

    async def upsert_chunks(self, chunks: List[ChunkVector]) -> None:
        if not chunks:
            return

        points: List[PointStruct] = []
        for chunk in chunks:
            vector: dict[str, Any] = {"dense": chunk.dense_vector}
            if chunk.sparse_indices:
                vector["bm25"] = SparseVector(
                    indices=chunk.sparse_indices, values=chunk.sparse_values
                )

            payload: dict[str, Any] = {
                "parent_chunk_id": chunk.parent_chunk_id,
                "doc_id": chunk.doc_id,
                "is_active": True,
                "breadcrumbs": chunk.breadcrumbs,
                "content_type": chunk.content_type,
                "text": chunk.text,
            }
            if chunk.session_id:
                payload["session_id"] = chunk.session_id

            points.append(PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            await self._client.upsert(
                collection_name=self.collection_name, points=points[i : i + batch_size]
            )

    async def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 15,
        session_id: Optional[str] = None,
        mode: str = "hybrid",
    ) -> List[SearchResult]:
        """Search the vector store using dense, sparse, or hybrid retrieval.

        Args:
            dense_vector: Dense embedding vector (BGE-M3, 1024-dim).
            sparse_indices: Sparse BM25 token indices.
            sparse_values: Sparse BM25 token weights.
            top_k: Maximum number of results to return.
            session_id: Optional session scope filter.
            mode: Retrieval mode — "hybrid" (RRF fusion), "dense" (dense only),
                or "sparse" (sparse/BM25 only).

        Returns:
            List of SearchResult ordered by relevance score.
        """
        if top_k <= 0:
            return []

        must_conditions: List[Any] = [FieldCondition(key="is_active", match=MatchValue(value=True))]
        if session_id:
            must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
        else:
            must_conditions.append(models.IsEmptyCondition(is_empty=models.PayloadField(key="session_id")))

        active_filter = Filter(must=must_conditions)

        if mode == "dense" or not sparse_indices:
            results = await self._client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                query_filter=active_filter,
                limit=top_k,
            )
        elif mode == "sparse":
            results = await self._client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="bm25",
                query_filter=active_filter,
                limit=top_k,
            )
        else:
            results = await self._client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="bm25", limit=top_k * 2, filter=active_filter
                    ),
                    models.Prefetch(
                        query=dense_vector, using="dense", limit=top_k * 2, filter=active_filter
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            )

        return [
            SearchResult(
                chunk_id=str(point.id),
                parent_chunk_id=str((point.payload or {}).get("parent_chunk_id", "")),
                doc_id=str((point.payload or {}).get("doc_id", "")),
                score=point.score,
            )
            for point in results.points
        ]

    async def update_payload(self, doc_id: str, payload: dict[str, Any]) -> None:
        await self._client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        )

    async def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise

    async def close(self) -> None:
        await self._client.close()
