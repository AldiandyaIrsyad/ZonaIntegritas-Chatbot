"""Tests for embedding provider infrastructure module."""

import httpx
import pytest
import respx

from app.infra.embedding_provider import EmbeddingProvider

@pytest.fixture
def provider():
    """Fixture providing an EmbeddingProvider instance."""
    return EmbeddingProvider(
        base_url="http://infinity:7997",
        model="BAAI/bge-m3",
        batch_size=2
    )

@respx.mock
async def test_embed_texts_success(provider):
    """Test successful embedding generation in batches."""
    # We send 3 texts, batch_size is 2. So two requests are expected.
    # First request: 2 texts
    respx.post("http://infinity:7997/embeddings").respond(
        json={
            "data": [
                {"index": 0, "embedding": [0.1, 0.2], "sparse_embedding": {"1": 0.5}},
                {"index": 1, "embedding": [0.3, 0.4], "sparse_embedding": {"2": 0.6}},
            ]
        }
    )
    
    results = await provider.embed_texts(["text1", "text2"])
    
    assert len(results) == 2
    assert results[0].dense == [0.1, 0.2]
    assert results[0].sparse_indices == [1]
    assert results[0].sparse_values == [0.5]
    
    assert results[1].dense == [0.3, 0.4]
    assert results[1].sparse_indices == [2]
    assert results[1].sparse_values == [0.6]

@respx.mock
async def test_embed_texts_order_guarantee(provider):
    """Test that embeddings are sorted by index from the response."""
    # The server might return out-of-order data
    respx.post("http://infinity:7997/embeddings").respond(
        json={
            "data": [
                {"index": 1, "embedding": [0.3, 0.4], "sparse_embedding": {}},
                {"index": 0, "embedding": [0.1, 0.2], "sparse_embedding": {}},
            ]
        }
    )
    
    results = await provider.embed_texts(["text1", "text2"])
    
    assert len(results) == 2
    assert results[0].dense == [0.1, 0.2]  # Index 0
    assert results[1].dense == [0.3, 0.4]  # Index 1

@respx.mock
async def test_embed_texts_size_mismatch(provider):
    """Test handling when response size doesn't match batch size."""
    # Send 2 texts, receive 1 result
    respx.post("http://infinity:7997/embeddings").respond(
        json={
            "data": [
                {"index": 0, "embedding": [0.1, 0.2], "sparse_embedding": {}}
            ]
        }
    )
    
    with pytest.raises(ValueError, match="Embedding response size mismatch"):
        await provider.embed_texts(["text1", "text2"])

@respx.mock
async def test_embed_texts_http_error(provider):
    """Test handling of HTTP errors."""
    respx.post("http://infinity:7997/embeddings").respond(status_code=400)
    
    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed_texts(["text1"])

async def test_embed_texts_empty(provider):
    """Test embedding an empty list of texts."""
    results = await provider.embed_texts([])
    assert results == []

async def test_close(provider):
    """Test closing the client."""
    await provider.close()
    assert provider._client.is_closed
