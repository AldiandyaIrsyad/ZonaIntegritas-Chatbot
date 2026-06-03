"""Qdrant vector database adapter.

Manages collection lifecycle, batch upserts with dense (bge-m3) + sparse
(BM25) vectors, hybrid RRF search, payload updates for toggling active state,
and cascade deletion by document ID.

Collection indexing strategy:
- Dense: cosine similarity, 1024 dimensions (BAAI/bge-m3)
- Sparse: BM25 with IDF modifier
- Payload indices: ``is_active`` (bool), ``session_id`` (keyword),
  ``doc_id`` (keyword) for fast pre-filtering
"""

from __future__ import annotations

from typing import Any, Optional

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    models,
)

from app.core.interfaces.infra import ChunkVector, IVectorStore, SearchResult

logger = structlog.get_logger(__name__)

# BAAI/bge-m3 produces 1024-dimensional dense vectors
BGE_M3_DENSE_DIM: int = 1024


class QdrantStore:
    """Async Qdrant adapter for hybrid dense + sparse vector search.

    Handles the full vector lifecycle: collection initialisation, batch
    upserts, hybrid RRF search with ``is_active`` pre-filtering, payload
    toggling, and cascade deletion.  Satisfies the
    :class:`~app.core.interfaces.infra.IVectorStore` Protocol structurally.

    Args:
        host: Qdrant server hostname (e.g. ``"qdrant"`` or ``"localhost"``).
        port: Qdrant REST/gRPC port (typically ``6333``).
        collection_name: Name of the Qdrant collection to manage.
    """

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self.collection_name = collection_name
        self._client = AsyncQdrantClient(host=host, port=port)
        logger.info(
            "QdrantStore initialised",
            host=host,
            port=port,
            collection_name=collection_name,
        )

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection and payload indices if absent.

        Safe to call on every application startup — exits early without error
        if the collection already exists.

        Raises:
            UnexpectedResponse: If Qdrant returns an unexpected HTTP error.
            Exception: Re-raised for any other Qdrant client error.
        """
        try:
            collections = await self._client.get_collections()
            existing = {c.name for c in collections.collections}

            if self.collection_name in existing:
                logger.debug(
                    "vector.collection.exists",
                    collection=self.collection_name,
                )
                return

            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=BGE_M3_DENSE_DIM,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

            for field_name, field_schema in [
                ("is_active", PayloadSchemaType.BOOL),
                ("session_id", PayloadSchemaType.KEYWORD),
                ("doc_id", PayloadSchemaType.KEYWORD),
            ]:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )

            logger.info(
                "vector.collection.created",
                collection=self.collection_name,
            )

        except UnexpectedResponse as exc:
            logger.error(
                "vector.collection.ensure_failed",
                collection=self.collection_name,
                status_code=exc.status_code,
                error=str(exc),
            )
            raise
        except Exception as exc:
            logger.error(
                "vector.collection.ensure_failed",
                collection=self.collection_name,
                error=str(exc),
            )
            raise

    async def upsert_chunks(self, chunks: list[ChunkVector]) -> None:
        """Batch-insert or update child chunk vectors.

        Validates dimension and index/value parity before building
        :class:`~qdrant_client.models.PointStruct` objects.  Upserted in
        batches of 100 to stay within Qdrant payload size limits.

        Args:
            chunks: Child chunk vectors to persist, including their dense/sparse
                    embeddings and metadata payloads.

        Raises:
            ValueError: If any chunk's ``dense_vector`` has wrong dimensions
                or ``sparse_indices``/``sparse_values`` lengths differ.
            Exception: Re-raised for any Qdrant client error.
        """
        if not chunks:
            return

        points: list[PointStruct] = []
        for chunk in chunks:
            if len(chunk.dense_vector) != BGE_M3_DENSE_DIM:
                raise ValueError(
                    f"dense_vector must be {BGE_M3_DENSE_DIM} dims, "
                    f"got {len(chunk.dense_vector)} for chunk_id={chunk.chunk_id!r}"
                )
            if len(chunk.sparse_indices) != len(chunk.sparse_values):
                raise ValueError(
                    f"sparse_indices/sparse_values length mismatch for "
                    f"chunk_id={chunk.chunk_id!r}"
                )

            vector: dict[str, Any] = {"dense": chunk.dense_vector}
            if chunk.sparse_indices:
                vector["bm25"] = SparseVector(
                    indices=chunk.sparse_indices,
                    values=chunk.sparse_values,
                )

            payload: dict[str, Any] = {
                "parent_chunk_id": chunk.parent_chunk_id,
                "doc_id": chunk.doc_id,
                "is_active": True,
            }
            if chunk.session_id:
                payload["session_id"] = chunk.session_id

            points.append(
                PointStruct(id=chunk.chunk_id, vector=vector, payload=payload)
            )

        batch_size = 100
        try:
            for i in range(0, len(points), batch_size):
                await self._client.upsert(
                    collection_name=self.collection_name,
                    points=points[i : i + batch_size],
                )
        except Exception as exc:
            logger.error(
                "vector.upsert.failed",
                collection=self.collection_name,
                chunk_count=len(chunks),
                error=str(exc),
            )
            raise

        logger.debug(
            "vector.upsert.complete",
            collection=self.collection_name,
            chunk_count=len(chunks),
            doc_id=chunks[0].doc_id,
            session_id=chunks[0].session_id,
        )

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 15,
        session_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Execute a hybrid dense + BM25 sparse search with RRF fusion.

        Pre-filters by ``is_active == True`` to respect document enable/disable
        state.  When ``session_id`` is given, only session-scoped documents are
        searched; otherwise only permanent (no session) documents are returned.

        Falls back to dense-only search if ``sparse_indices`` is empty.

        Args:
            dense_vector: Dense query embedding
                          (must be exactly ``BGE_M3_DENSE_DIM`` dimensions).
            sparse_indices: BM25 sparse token indices for the query.
            sparse_values: Corresponding BM25 weights.
            top_k: Maximum number of results to return.  Returns ``[]`` if
                   ``<= 0``.
            session_id: If provided, restricts search to this session scope.

        Returns:
            Up to ``top_k`` :class:`~app.core.interfaces.infra.SearchResult`
            items ranked by RRF fusion score.

        Raises:
            ValueError: If vector dimensions or sparse array lengths are invalid.
            Exception: Re-raised for any Qdrant client error.
        """
        if top_k <= 0:
            return []
        if len(dense_vector) != BGE_M3_DENSE_DIM:
            raise ValueError(
                f"dense_vector must be {BGE_M3_DENSE_DIM} dims, "
                f"got {len(dense_vector)}"
            )
        if len(sparse_indices) != len(sparse_values):
            raise ValueError(
                "sparse_indices and sparse_values must have the same length"
            )

        must_conditions: list[Any] = [
            FieldCondition(key="is_active", match=MatchValue(value=True))
        ]
        if session_id:
            must_conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )
        else:
            must_conditions.append(
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="session_id")
                )
            )

        active_filter = Filter(must=must_conditions)  # type: ignore[arg-type]

        try:
            if not sparse_indices:
                # Fallback: dense-only search when BM25 indices are absent
                results = await self._client.query_points(
                    collection_name=self.collection_name,
                    query=dense_vector,
                    using="dense",
                    query_filter=active_filter,
                    limit=top_k,
                )
            else:
                # RRF hybrid: BM25 + dense prefetch, fused with Reciprocal Rank Fusion
                results = await self._client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            ),
                            using="bm25",
                            limit=top_k * 2,
                            filter=active_filter,
                        ),
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=top_k * 2,
                            filter=active_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                )
        except Exception as exc:
            logger.error(
                "vector.search.failed",
                collection=self.collection_name,
                top_k=top_k,
                session_id=session_id,
                error=str(exc),
            )
            raise

        search_results = [
            SearchResult(
                chunk_id=str(point.id),
                parent_chunk_id=str(point.payload.get("parent_chunk_id", "")),
                doc_id=str(point.payload.get("doc_id", "")),
                score=point.score,
            )
            for point in results.points
        ]

        logger.debug(
            "vector.search.complete",
            collection=self.collection_name,
            top_k=top_k,
            result_count=len(search_results),
            session_id=session_id,
        )
        return search_results

    async def update_payload(self, doc_id: str, payload: dict[str, Any]) -> None:
        """Update metadata payload fields for all vectors of a document.

        Typically used to toggle the ``is_active`` flag without re-embedding.

        Args:
            doc_id: UUID of the target document.
            payload: Key-value pairs to set on all matching vector points.

        Raises:
            Exception: Re-raised for any Qdrant client error.
        """
        try:
            await self._client.set_payload(
                collection_name=self.collection_name,
                payload=payload,  # type: ignore[arg-type]
                points=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )
        except Exception as exc:
            logger.error(
                "vector.payload.update_failed",
                doc_id=doc_id,
                fields=list(payload.keys()),
                error=str(exc),
            )
            raise

        logger.debug(
            "vector.payload.updated",
            doc_id=doc_id,
            fields=list(payload.keys()),
        )

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors associated with a document.

        Args:
            doc_id: UUID of the document whose vectors should be removed.

        Raises:
            Exception: Re-raised for any Qdrant client error.
        """
        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                logger.debug("vector.delete.skipped_not_found", doc_id=doc_id)
                return
            logger.error(
                "vector.delete.failed",
                doc_id=doc_id,
                error=str(exc),
            )
            raise
        except Exception as exc:
            logger.error(
                "vector.delete.failed",
                doc_id=doc_id,
                error=str(exc),
            )
            raise

        logger.debug("vector.delete.complete", doc_id=doc_id)

    async def close(self) -> None:
        """Close the Qdrant async client connection."""
        await self._client.close()
