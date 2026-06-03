"""Tests for vector store infrastructure module."""

import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, models

from app.infra.vector_store import QdrantStore, BGE_M3_DENSE_DIM
from app.core.interfaces.infra import ChunkVector

@pytest.fixture
def mock_qdrant_client():
    """Fixture providing a mock AsyncQdrantClient."""
    with patch("app.infra.vector_store.AsyncQdrantClient") as mock_client_class:
        mock_instance = AsyncMock()
        mock_client_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def vector_store(mock_qdrant_client):
    """Fixture providing a QdrantStore instance."""
    return QdrantStore(
        host="localhost",
        port=6333,
        collection_name="test_collection"
    )

async def test_ensure_collection_exists(vector_store, mock_qdrant_client):
    """Test when collection already exists."""
    mock_collections = MagicMock()
    mock_collections.collections = [MagicMock(name="test_collection")]
    # Because of MagicMock name handling, we need to explicitly set the attribute name
    mock_collections.collections[0].name = "test_collection"
    
    mock_qdrant_client.get_collections.return_value = mock_collections
    
    await vector_store.ensure_collection()
    
    mock_qdrant_client.create_collection.assert_not_awaited()

async def test_ensure_collection_creates(vector_store, mock_qdrant_client):
    """Test creating a new collection."""
    mock_collections = MagicMock()
    mock_collections.collections = []
    
    mock_qdrant_client.get_collections.return_value = mock_collections
    
    await vector_store.ensure_collection()
    
    mock_qdrant_client.create_collection.assert_awaited_once_with(
        collection_name="test_collection",
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
    
    assert mock_qdrant_client.create_payload_index.await_count == 3

async def test_upsert_chunks_success(vector_store, mock_qdrant_client):
    """Test successful upsert of valid chunks."""
    chunks = [
        ChunkVector(
            chunk_id=str(uuid.uuid4()),
            parent_chunk_id="parent1",
            doc_id="doc1",
            dense_vector=[0.1] * BGE_M3_DENSE_DIM,
            sparse_indices=[1, 2, 3],
            sparse_values=[0.5, 0.6, 0.7],
            session_id="session1"
        )
    ]
    
    await vector_store.upsert_chunks(chunks)
    
    mock_qdrant_client.upsert.assert_awaited_once()

async def test_upsert_chunks_invalid_dense_dim(vector_store):
    """Test upsert fails on invalid dense vector dimensions."""
    chunks = [
        ChunkVector(
            chunk_id=str(uuid.uuid4()),
            parent_chunk_id="parent1",
            doc_id="doc1",
            dense_vector=[0.1] * (BGE_M3_DENSE_DIM - 1),  # Invalid dim
            sparse_indices=[],
            sparse_values=[]
        )
    ]
    
    with pytest.raises(ValueError, match="dense_vector must be"):
        await vector_store.upsert_chunks(chunks)

async def test_upsert_chunks_sparse_mismatch(vector_store):
    """Test upsert fails on sparse indices/values mismatch."""
    chunks = [
        ChunkVector(
            chunk_id=str(uuid.uuid4()),
            parent_chunk_id="parent1",
            doc_id="doc1",
            dense_vector=[0.1] * BGE_M3_DENSE_DIM,
            sparse_indices=[1, 2],
            sparse_values=[0.5]  # Mismatch
        )
    ]
    
    with pytest.raises(ValueError, match="sparse_indices/sparse_values length mismatch"):
        await vector_store.upsert_chunks(chunks)

async def test_hybrid_search_success(vector_store, mock_qdrant_client):
    """Test hybrid search with RRF."""
    # Setup mock response
    mock_response = MagicMock()
    point1 = MagicMock()
    point1.id = "id1"
    point1.payload = {"parent_chunk_id": "p1", "doc_id": "d1"}
    point1.score = 0.9
    mock_response.points = [point1]
    
    mock_qdrant_client.query_points.return_value = mock_response
    
    results = await vector_store.hybrid_search(
        dense_vector=[0.1] * BGE_M3_DENSE_DIM,
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.6],
        top_k=5,
        session_id="test_session"
    )
    
    assert len(results) == 1
    assert results[0].chunk_id == "id1"
    assert results[0].doc_id == "d1"
    assert results[0].score == 0.9
    
    # Assert query points used RRF fusion
    args, kwargs = mock_qdrant_client.query_points.call_args
    assert "prefetch" in kwargs
    assert kwargs["limit"] == 5

async def test_hybrid_search_dense_fallback(vector_store, mock_qdrant_client):
    """Test fallback to dense search when sparse indices are empty."""
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant_client.query_points.return_value = mock_response
    
    await vector_store.hybrid_search(
        dense_vector=[0.1] * BGE_M3_DENSE_DIM,
        sparse_indices=[],
        sparse_values=[],
        top_k=5
    )
    
    args, kwargs = mock_qdrant_client.query_points.call_args
    assert "prefetch" not in kwargs
    assert kwargs["using"] == "dense"

async def test_update_payload(vector_store, mock_qdrant_client):
    """Test payload update."""
    await vector_store.update_payload(
        doc_id="doc1",
        payload={"is_active": False}
    )
    
    mock_qdrant_client.set_payload.assert_awaited_once()

async def test_delete_by_doc_id(vector_store, mock_qdrant_client):
    """Test cascade deletion by doc id."""
    await vector_store.delete_by_doc_id("doc1")
    
    mock_qdrant_client.delete.assert_awaited_once()

async def test_close(vector_store, mock_qdrant_client):
    """Test closing the connection."""
    await vector_store.close()
    mock_qdrant_client.close.assert_awaited_once()
