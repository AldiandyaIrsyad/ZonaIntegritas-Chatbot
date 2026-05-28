"""
Qdrant vector database client wrapper.

Handles connection management, collection initialization with hybrid search
(dense + BM25 sparse vectors), and provides methods for upserting, searching,
and managing vector payloads.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    ScoredPoint,
    SetPayloadOperation,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    PayloadSchemaType,
    models,
)

from src.core.logging import get_logger
from src.core.events import LogEvent

logger = get_logger(__name__)


# BGE-M3 produces 1024-dimensional dense vectors
BGE_M3_DENSE_DIM = 1024


@dataclass
class ChunkVector:
    """Represents a child chunk with its dense and sparse embeddings."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]
    session_id: Optional[str] = None


@dataclass
class SearchResult:
    """A single result from hybrid search."""
    chunk_id: str
    parent_chunk_id: str
    doc_id: str
    score: float


class QdrantStore:
    """
    Abstraction layer for the Qdrant vector database.

    Handles:
    - Collection creation with dense (bge-m3) + sparse (BM25) vectors
    - Batch upserts with doc/parent metadata payloads
    - Hybrid search with mandatory is_active filtering
    - Payload updates for toggling document active state
    - Cascade deletion by doc_id
    """

    def __init__(self, host: str, port: int, collection_name: str):
        self.collection_name = collection_name
        self.client = AsyncQdrantClient(host=host, port=port)

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist.

        Configures both dense vectors (for semantic similarity via bge-m3)
        and sparse vectors (for BM25 lexical matching with IDF modifier).
        """
        collections = await self.client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if self.collection_name not in existing_names:
            await self.client.create_collection(
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

            # Create payload index for fast filtering on is_active and doc_id
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="is_active",
                field_schema=PayloadSchemaType.BOOL,
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(
                "Created Qdrant collection '%s' with dense + sparse vectors",
                self.collection_name,
            )
        else:
            logger.info(
                "Qdrant collection '%s' already exists, skipping creation",
                self.collection_name,
            )

    async def upsert_chunks(self, chunks: List[ChunkVector]) -> None:
        """Batch upsert child chunk vectors with parent/doc metadata.

        Each point carries a payload with:
        - parent_chunk_id: reference to the parent chunk in PostgreSQL
        - doc_id: reference to the source PDFDocument
        - is_active: whether this document is currently active for retrieval
        """
        if not chunks:
            return

        points = []
        for chunk in chunks:
            if len(chunk.dense_vector) != BGE_M3_DENSE_DIM:
                raise ValueError(
                    f"dense_vector must have {BGE_M3_DENSE_DIM} dimensions, "
                    f"got {len(chunk.dense_vector)} for chunk_id={chunk.chunk_id}"
                )
            if len(chunk.sparse_indices) != len(chunk.sparse_values):
                raise ValueError(
                    f"sparse_indices and sparse_values length mismatch for "
                    f"chunk_id={chunk.chunk_id}"
                )
            point = PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense": chunk.dense_vector,
                },
                payload={
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "doc_id": chunk.doc_id,
                    "is_active": True,
                }
            )
            if chunk.session_id:
                point.payload["session_id"] = chunk.session_id
            # Attach sparse vector separately if it exists
            if chunk.sparse_indices:
                point.vector["bm25"] = SparseVector(
                    indices=chunk.sparse_indices,
                    values=chunk.sparse_values,
                )
            points.append(point)

        # Upsert in batches of 100 to avoid payload size limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info("Upserted %d chunk vectors to Qdrant", len(chunks), extra={
            "event": LogEvent.VECTOR_UPSERT.value,
            "doc_id": chunks[0].doc_id if chunks else None,
            "session_id": chunks[0].session_id if chunks else None,
            "count": len(chunks),
            "payload_sample": [p.payload for p in points[:3]] if points else []
        })

    async def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 15,
        session_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute hybrid search combining dense semantic and sparse BM25 matching.

        Only searches vectors where is_active == true to respect document
        enable/disable state managed via the admin dashboard.
        """
        if top_k <= 0:
            return []
        if len(dense_vector) != BGE_M3_DENSE_DIM:
            raise ValueError(
                f"dense_vector must have {BGE_M3_DENSE_DIM} dimensions, got {len(dense_vector)}"
            )
        if len(sparse_indices) != len(sparse_values):
            raise ValueError("sparse_indices and sparse_values must have the same length")
        must_conditions = [
            FieldCondition(
                key="is_active",
                match=MatchValue(value=True),
            )
        ]
        if session_id:
            must_conditions.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            )

        active_filter = Filter(must=must_conditions)

        if not sparse_indices:
            # Fallback to standard dense search if sparse vectors are missing
            results = await self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                query_filter=active_filter,
                limit=top_k,
            )
        else:
            # Execute RRF hybrid search
            results = await self.client.query_points(
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

        return [
            SearchResult(
                chunk_id=str(point.id),
                parent_chunk_id=point.payload.get("parent_chunk_id", ""),
                doc_id=point.payload.get("doc_id", ""),
                score=point.score,
            )
            for point in results.points
        ]

    async def update_payload(
        self, doc_id: str, payload: dict
    ) -> None:
        """Update payload fields for all vectors belonging to a document.

        Used to toggle is_active state without re-embedding.
        """
        await self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        logger.info(
            "Updated payload for doc_id='%s': %s", doc_id, payload
        )

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors associated with a document.

        Called when a PDF is permanently deleted from the knowledge base.
        """
        await self.client.delete(
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
        logger.info("Deleted all vectors for doc_id='%s'", doc_id)

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self.client.close()
